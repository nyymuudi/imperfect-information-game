"""Sanity tests for SubgameDistiller.

These run on a small fake blueprint at 50bb / 1-raise and validate that:

  * `measure_blueprint_gap` returns a non-trivial GapReport on a random
    blueprint (TV > 0, argmax-disagreement > 0).
  * `generate_distillation_targets` returns valid (state, strategy) pairs
    with the right shapes and probability-simplex strategies.
  * Repeating with the same seed gives identical results.

The aim is to surface algorithmic regressions (e.g. solver returning
uniform strategies, range construction errors) without paying for a
full blueprint training run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE
from src.solvers.subgame_distillation import (
    DistillationTarget,
    GapReport,
    SubgameDistiller,
    _uniform_range,
)


# ── Lightweight blueprint stand-in ───────────────────────────────────────────


class _Meta:
    def __init__(self, state_size: int, action_size: int = 4):
        self.state_size     = state_size
        self.action_size    = action_size
        self.starting_stack = 50.0
        self.max_raises     = 1
        self.raise_fraction = 0.5
        self.raise_fractions: list = []


class _RandomBlueprint:
    """Per-state pseudo-random distribution. Deterministic by hash so two
    queries on the same state vector return the same probs (matching real
    blueprint determinism)."""

    def __init__(self, state_size: int):
        self.metadata = _Meta(state_size)

    def query(self, state_vec, num_actions: int) -> np.ndarray:
        seed = int(np.abs(state_vec.astype(np.float64)).sum() * 1e6) % (2**31)
        rng = np.random.default_rng(seed)
        p = rng.random(num_actions) + 1e-3
        return p / p.sum()

    def query_by_slots(self, state_vec, slot_indices) -> np.ndarray:
        # Deterministic by state + slot signature so seed-reproducibility
        # holds (network slot ≠ legal-action index, but the test only
        # needs sums and shapes to behave consistently).
        seed = int(np.abs(state_vec.astype(np.float64)).sum() * 1e6) % (2**31)
        rng  = np.random.default_rng(seed)
        p    = rng.random(len(slot_indices)) + 1e-3
        return p / p.sum()


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def game():
    return PostflopNLHE(
        starting_stack=50.0,
        max_raises_per_street=1,
        raise_fractions=(0.5,),
    )


@pytest.fixture(scope="module")
def encoder():
    NLHEEncoder._shared_equity_cache = None
    return NLHEEncoder(starting_stack=50.0, equity_sims=20)


@pytest.fixture(scope="module")
def distiller(game, encoder):
    bp = _RandomBlueprint(encoder.state_size())
    return SubgameDistiller(
        blueprint=bp,
        encoder=encoder,
        base_game=game,
        iterations=50,
        max_deals=12,
        max_walk_depth=4,
    )


# ── Range helper ─────────────────────────────────────────────────────────────


class TestUniformRange:

    def test_size_matches_combinations(self):
        # 52 - 5 board = 47 live cards; C(47, 2) = 1081
        r = _uniform_range(board=(0, 5, 10, 15, 20))
        assert len(r) == 1081
        assert sum(r.values()) == pytest.approx(1.0)

    def test_excludes_hero_cards(self):
        r = _uniform_range(board=(0, 5, 10, 15, 20), exclude=(48, 49))
        # 52 - 5 - 2 = 45 → C(45, 2) = 990
        assert len(r) == 990

    def test_no_pair_contains_board_cards(self):
        board = (0, 5, 10, 15, 20)
        r = _uniform_range(board=board)
        for (a, b) in r.keys():
            assert a not in board and b not in board


# ── Diagnostic API ───────────────────────────────────────────────────────────


class TestMeasureBlueprintGap:

    def test_returns_gap_report(self, distiller):
        rep = distiller.measure_blueprint_gap(n_spots=8, seed=0)
        assert isinstance(rep, GapReport)
        # On a random blueprint, almost-sure ‖blueprint − solver‖ > 0
        assert rep.mean_tv > 0.0
        assert rep.n_spots > 0

    def test_seed_reproducibility(self, distiller):
        a = distiller.measure_blueprint_gap(n_spots=5, seed=42)
        b = distiller.measure_blueprint_gap(n_spots=5, seed=42)
        assert a.mean_tv == pytest.approx(b.mean_tv, rel=1e-6)

    def test_argmax_disagreement_in_unit_interval(self, distiller):
        rep = distiller.measure_blueprint_gap(n_spots=8, seed=1)
        assert 0.0 <= rep.mean_argmax_disagreement <= 1.0


# ── Distillation target API ──────────────────────────────────────────────────


class TestGenerateDistillationTargets:

    def test_targets_have_correct_shapes(self, distiller, encoder):
        targets = distiller.generate_distillation_targets(n_spots=8, seed=2)
        assert len(targets) > 0
        action_size = distiller.blueprint.metadata.action_size
        for t in targets:
            assert isinstance(t, DistillationTarget)
            assert t.state.shape == (encoder.state_size(),)
            # Slot-indexed: length equals network action_size, not n_legal.
            assert t.strategy.shape == (action_size,)
            assert len(t.legal_slots) > 0
            assert all(0 <= s < action_size for s in t.legal_slots)

    def test_strategies_are_probability_simplex(self, distiller):
        targets = distiller.generate_distillation_targets(n_spots=8, seed=3)
        for t in targets:
            assert t.strategy.min() >= -1e-6
            assert t.strategy.sum() == pytest.approx(1.0, abs=1e-3)

    def test_seed_reproducibility(self, distiller):
        a = distiller.generate_distillation_targets(n_spots=5, seed=7)
        b = distiller.generate_distillation_targets(n_spots=5, seed=7)
        assert len(a) == len(b)
        for t_a, t_b in zip(a, b):
            np.testing.assert_allclose(t_a.state,    t_b.state,    atol=1e-6)
            np.testing.assert_allclose(t_a.strategy, t_b.strategy, atol=1e-6)


# ── Targeted scenario sampler ────────────────────────────────────────────────


class TestTargetedScenarios:

    def test_invalid_scenario_rejected(self, game, encoder):
        from src.solvers.subgame_distillation import SubgameDistiller
        bp = _RandomBlueprint(encoder.state_size())
        with pytest.raises(ValueError):
            SubgameDistiller(
                blueprint=bp, encoder=encoder, base_game=game,
                scenarios=("not_a_real_scenario",),
            )

    def test_preflop_sb_lands_4_actions(self, game, encoder):
        """preflop_sb scenario should produce a state with 4 legal actions
        (fold/call/raise/allin) — SB at the initial decision facing BB blind.
        """
        from src.solvers.subgame_distillation import SubgameDistiller
        bp = _RandomBlueprint(encoder.state_size())
        d = SubgameDistiller(
            blueprint=bp, encoder=encoder, base_game=game,
            scenarios=("preflop_sb",),
        )
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(20):
            root = d._construct_scenario("preflop_sb", rng)
            if root is None:
                continue
            legal = game.legal_actions(root)
            assert len(legal) == 4, f"expected 4 actions, got {legal}"
            assert "f" in legal and "k" in legal \
                   and "r" in legal and "a" in legal
            assert game.current_player(root) == 0  # SB
            ok += 1
        assert ok >= 18, f"preflop_sb produced too many None roots ({ok}/20)"

    def test_preflop_bb_facing_lands_face_bet(self, game, encoder):
        """preflop_bb_facing: SB raises preflop → BB faces. BB to act with
        to_call > 0. Action count is 4 (f/k/r/a) when max_raises ≥ 2,
        otherwise 3 (f/k/a) — the raise counter caps the available options.
        """
        from src.solvers.subgame_distillation import SubgameDistiller
        bp = _RandomBlueprint(encoder.state_size())
        d = SubgameDistiller(
            blueprint=bp, encoder=encoder, base_game=game,
            scenarios=("preflop_bb_facing",),
        )
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(20):
            root = d._construct_scenario("preflop_bb_facing", rng)
            if root is None:
                continue
            legal = game.legal_actions(root)
            # Always includes fold + call when facing a bet, plus all-in.
            assert "f" in legal and "k" in legal and "a" in legal
            assert game.current_player(root) == 1  # BB
            ok += 1
        assert ok >= 18

    def test_flop_no_bet_lands_3_actions(self, game, encoder):
        """flop_no_bet: SB call → BB check → SB on flop. 3-action (c/r/a)."""
        from src.solvers.subgame_distillation import SubgameDistiller
        bp = _RandomBlueprint(encoder.state_size())
        d = SubgameDistiller(
            blueprint=bp, encoder=encoder, base_game=game,
            scenarios=("flop_no_bet",),
        )
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(20):
            root = d._construct_scenario("flop_no_bet", rng)
            if root is None:
                continue
            legal = game.legal_actions(root)
            assert "c" in legal and "r" in legal and "a" in legal
            assert "k" not in legal  # no call (no bet to call)
            assert game.current_player(root) == 0  # SB
            ok += 1
        assert ok >= 18

    def test_scenarios_rotate_uniformly(self, game, encoder):
        """When >1 scenario is configured, generate_distillation_targets
        should rotate through them in round-robin fashion."""
        from src.solvers.subgame_distillation import SubgameDistiller
        bp = _RandomBlueprint(encoder.state_size())
        # Use cheap config: small iter/deals to keep test fast.
        d = SubgameDistiller(
            blueprint=bp, encoder=encoder, base_game=game,
            iterations=20, max_deals=8, max_walk_depth=3,
            scenarios=("preflop_sb", "flop_no_bet"),
        )
        # Patch solve_one to a fast no-op for this rotation test — we only
        # care that _construct_scenario gets called with alternating names.
        called = []
        original_construct = d._construct_scenario
        def spy(scenario, rng):
            called.append(scenario)
            return original_construct(scenario, rng)
        d._construct_scenario = spy

        # Patch solve_one so we don't burn CPU on tabular CFR here.
        def fake_solve(root, rng):
            from src.solvers.subgame_solver import SubgameStrategy, SubgameGame
            sg = SubgameGame(
                base_game=game, root_history=root, hero_player=0,
                hero_range={(0, 1): 1.0}, opponent_range={(2, 3): 1.0},
                max_deals=1, rng=rng,
            )
            return SubgameStrategy({}, sg)
        d.solve_one = fake_solve

        d.generate_distillation_targets(n_spots=10, seed=0)
        # Round-robin: first 5 should alternate scenarios.
        assert called[0] == "preflop_sb"
        assert called[1] == "flop_no_bet"
        assert called[2] == "preflop_sb"
        assert called[3] == "flop_no_bet"
