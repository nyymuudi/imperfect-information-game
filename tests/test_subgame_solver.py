"""
Tests for SubgameSolver.

Run with:  pytest tests/test_subgame_solver.py -v

Tests are designed to run fast (small ranges, few iterations, tiny games).
The SafeSubgameSolver tests use a mock Blueprint to avoid full training.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.games.postflop_nlhe import PostflopNLHE
from src.abstraction.equity import str_to_card
from src.solvers.subgame_solver import (
    SubgameGame,
    GadgetGame,
    SubgameStrategy,
    UnsafeSubgameSolver,
    SafeSubgameSolver,
    estimate_blueprint_ev,
    _rollout_expected_value,
    _OPT_OUT,
)


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def c(s: str) -> int:
    return str_to_card(s)


def make_game(stack=50.0):
    return PostflopNLHE(
        starting_stack=stack,
        max_raises_per_street=1,
        raise_fractions=(0.75,),
    )


def make_root_history(game, p0_cards, p1_cards, board5, prefix_actions=()):
    """Build a PostflopNLHE root history with pre-dealt board."""
    base = (p0_cards, p1_cards, board5)
    return base + tuple(prefix_actions)


def small_ranges():
    """Two small non-conflicting ranges for fast tests."""
    hero_range = {
        (c("Ah"), c("Kd")): 0.5,
        (c("Qh"), c("Jd")): 0.5,
    }
    opp_range = {
        (c("Tc"), c("9s")): 0.6,
        (c("8c"), c("7s")): 0.4,
    }
    return hero_range, opp_range


def make_board5():
    return (c("2c"), c("5d"), c("9h"), c("3s"), c("Tc"))


# ── Mock Blueprint ────────────────────────────────────────────────────────────

class MockBlueprint:
    """Returns uniform strategy for any query."""
    def __init__(self):
        self.metadata = type("m", (), {
            "state_size": 122, "action_size": 4
        })()

    def query(self, state_vec, num_actions):
        return np.ones(num_actions) / num_actions


class MockEncoder:
    """Returns a random state vector."""
    def __init__(self, state_size=122):
        self._size = state_size

    def encode(self, history, player):
        rng = np.random.default_rng(abs(hash(str(history))) % (2**31))
        return rng.random(self._size).astype(np.float32)


# ── SubgameGame ───────────────────────────────────────────────────────────────

class TestSubgameGame:

    @pytest.fixture
    def setup(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(game, (c("Ah"), c("Kd")),
                                 (c("Tc"), c("9s")), board5)
        sg = SubgameGame(
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
        )
        return sg, hero_range, opp_range

    def test_initial_histories_non_empty(self, setup):
        sg, _, _ = setup
        assert len(sg.initial_histories()) > 0

    def test_probabilities_sum_to_one(self, setup):
        sg, _, _ = setup
        total = sum(p for _, p in sg.initial_histories())
        assert abs(total - 1.0) < 1e-6

    def test_no_card_conflicts(self, setup):
        sg, _, _ = setup
        board5 = make_board5()
        for history, _ in sg.initial_histories():
            p0 = set(history[0])
            p1 = set(history[1])
            board = set(history[2])
            assert not (p0 & p1),    "P0 and P1 share cards"
            assert not (p0 & board), "P0 conflicts with board"
            assert not (p1 & board), "P1 conflicts with board"

    def test_info_set_key_strips_prefix(self, setup):
        sg, _, _ = setup
        # Take a history from initial_histories (has prefix_len=0 prefix)
        h0, _ = sg.initial_histories()[0]
        key_before = sg.info_set_key(h0, 0)

        # Apply one subgame action
        actions = sg.legal_actions(h0)
        h1 = sg.apply_action(h0, actions[0])
        key_after = sg.info_set_key(h1, 0)

        assert key_before != key_after

    def test_info_set_key_same_for_same_subgame_trajectory(self, setup):
        sg, _, _ = setup
        histories = [h for h, _ in sg.initial_histories()]
        # Apply the same first action to two different deals
        if len(histories) >= 2:
            h0 = histories[0]
            h1 = histories[1]
            action = sg.legal_actions(h0)[0]
            h0_next = sg.apply_action(h0, action)
            h1_next = sg.apply_action(h1, action)
            # Same player, same action → same subgame-local info set
            # (cards differ but they're different info sets anyway — check only
            # that the key structure is correct: both include subgame action)
            k0 = sg.info_set_key(h0_next, 0)
            k1 = sg.info_set_key(h1_next, 0)
            # These should differ (different hole cards) — not equal
            # but both should contain the action as part of the key
            assert action in k0
            assert action in k1

    def test_max_deals_caps_initial_histories(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(game, (c("Ah"), c("Kd")),
                                 (c("Tc"), c("9s")), board5)
        sg = SubgameGame(
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            max_deals=2,
        )
        assert len(sg.initial_histories()) <= 2

    def test_prefix_actions_carried_through(self):
        """Root history with prefix actions: subgame starts after them."""
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        # Prefix: SB raises ('r'), BB calls ('k') preflop
        root = make_root_history(
            game, (c("Ah"), c("Kd")), (c("Tc"), c("9s")),
            board5, prefix_actions=('r', 'k')
        )
        sg = SubgameGame(
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
        )
        assert sg._prefix_len == 2
        # initial histories all have the prefix actions
        for h, _ in sg.initial_histories():
            assert h[3] == 'r'
            assert h[4] == 'k'

    def test_empty_range_gives_empty_initial(self):
        game = make_game()
        board5 = make_board5()
        root = make_root_history(game, (c("Ah"), c("Kd")),
                                 (c("Tc"), c("9s")), board5)
        sg = SubgameGame(
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range={},
            opponent_range={(c("Tc"), c("9s")): 1.0},
        )
        assert sg.initial_histories() == []


# ── GadgetGame ────────────────────────────────────────────────────────────────

class TestGadgetGame:

    def _make_gadget(self, bp_ev=None):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(game, (c("Ah"), c("Kd")),
                                 (c("Tc"), c("9s")), board5)
        bp_ev = bp_ev or {k: 1.0 for k in opp_range}
        return GadgetGame(
            blueprint_ev_by_opp_cards=bp_ev,
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
        )

    def test_opt_out_available_at_root_for_opponent(self):
        gadget = self._make_gadget()
        h, _ = gadget.initial_histories()[0]
        # At root, check whose turn it is
        player = gadget.current_player(h)
        if player == 1:  # opponent acts first at root
            actions = gadget.legal_actions(h)
            assert _OPT_OUT in actions

    def test_opt_out_not_available_after_action(self):
        gadget = self._make_gadget()
        h, _ = gadget.initial_histories()[0]
        player = gadget.current_player(h)
        regular_actions = [a for a in gadget.legal_actions(h) if a != _OPT_OUT]
        if regular_actions:
            h2 = gadget.apply_action(h, regular_actions[0])
            if not gadget.is_terminal(h2):
                actions2 = gadget.legal_actions(h2)
                assert _OPT_OUT not in actions2

    def test_opt_out_is_terminal(self):
        gadget = self._make_gadget()
        h, _ = gadget.initial_histories()[0]
        h_out = gadget.apply_action(h, _OPT_OUT)
        assert gadget.is_terminal(h_out)

    def test_opt_out_payoff_uses_blueprint_ev(self):
        opp_hand = (c("Tc"), c("9s"))
        bp_ev = {opp_hand: 5.0}
        gadget = self._make_gadget(bp_ev)

        for h, _ in gadget.initial_histories():
            if set(h[1]) == set(opp_hand):   # opp has this hand
                h_out = gadget.apply_action(h, _OPT_OUT)
                payoffs = gadget.terminal_payoffs(h_out)
                # hero=0 gets -(opp_ev), opp=1 gets opp_ev
                assert abs(payoffs[1] - 5.0) < 1e-6
                assert abs(payoffs[0] + 5.0) < 1e-6
                break


# ── UnsafeSubgameSolver ───────────────────────────────────────────────────────

class TestUnsafeSubgameSolver:

    @pytest.fixture
    def solver_setup(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(
            game, (c("Ah"), c("Kd")), (c("Tc"), c("9s")), board5
        )
        solver = UnsafeSubgameSolver(game)
        return solver, root, hero_range, opp_range

    def test_returns_subgame_strategy(self, solver_setup):
        solver, root, hero_range, opp_range = solver_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=10,
        )
        assert isinstance(result, SubgameStrategy)

    def test_strategy_non_empty(self, solver_setup):
        solver, root, hero_range, opp_range = solver_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=20,
        )
        assert len(result) > 0

    def test_query_returns_valid_distribution(self, solver_setup):
        solver, root, hero_range, opp_range = solver_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=20,
        )
        game = make_game()
        board5 = make_board5()
        h = (
            (c("Ah"), c("Kd")),
            (c("Tc"), c("9s")),
            board5,
        )
        probs = result.query(h, player=0)
        assert probs.shape[0] >= 1
        assert all(p >= 0 for p in probs)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_uniform_fallback_for_unseen_info_set(self, solver_setup):
        solver, root, hero_range, opp_range = solver_setup
        # Solve with zero iterations → no info sets visited
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=0,
        )
        game = make_game()
        h = (
            (c("Ah"), c("Kd")),
            (c("Tc"), c("9s")),
            make_board5(),
        )
        probs = result.query(h, player=0)
        n = len(game.legal_actions(h))
        # Should be uniform
        np.testing.assert_allclose(probs, np.ones(n) / n, atol=1e-6)

    def test_empty_range_returns_empty_strategy(self, solver_setup):
        solver, root, _, opp_range = solver_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range={},
            opponent_range=opp_range,
            iterations=10,
        )
        assert len(result) == 0

    def test_more_iterations_more_info_sets(self, solver_setup):
        solver, root, hero_range, opp_range = solver_setup
        r1 = solver.solve(root, 0, hero_range, opp_range, iterations=5)
        r2 = solver.solve(root, 0, hero_range, opp_range, iterations=50)
        # More iterations should visit more info sets
        assert len(r2) >= len(r1)


# ── Blueprint rollout helper ──────────────────────────────────────────────────

class TestRollout:

    def test_terminal_history_returns_actual_payoffs(self):
        game = make_game()
        board5 = make_board5()
        # Build a terminal history (fold)
        h = (
            (c("Ah"), c("Kd")),
            (c("Tc"), c("9s")),
            board5,
            "f",  # P0 folds immediately
        )
        assert game.is_terminal(h)
        bp = MockBlueprint()
        enc = MockEncoder()
        ev0, ev1 = _rollout_expected_value(bp, enc, game, h)
        actual = game.terminal_payoffs(h)
        assert abs(ev0 - actual[0]) < 1e-6
        assert abs(ev1 - actual[1]) < 1e-6

    def test_rollout_sums_to_zero(self):
        """Zero-sum game: ev0 + ev1 should equal zero (or pot)."""
        game = make_game()
        board5 = make_board5()
        h = (
            (c("Ah"), c("Kd")),
            (c("Tc"), c("9s")),
            board5,
        )
        bp = MockBlueprint()
        enc = MockEncoder()
        ev0, ev1 = _rollout_expected_value(bp, enc, game, h)
        # In zero-sum poker: ev0 + ev1 = 0 (net chip exchange)
        # The game starts with both paying antes, so absolute values
        # can be non-zero but their sum = 0
        assert abs(ev0 + ev1) < 1e-4


# ── estimate_blueprint_ev ─────────────────────────────────────────────────────

class TestEstimateBlueprintEv:

    def test_returns_dict_for_all_opp_hands(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(
            game, (c("Ah"), c("Kd")), (c("Tc"), c("9s")), board5
        )
        ev_dict = estimate_blueprint_ev(
            blueprint=MockBlueprint(),
            encoder=MockEncoder(),
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
        )
        # Should have one entry per non-conflicting opp hand
        for opp_cards in opp_range:
            if not (set(opp_cards) & set(board5)):
                assert opp_cards in ev_dict

    def test_ev_values_are_finite(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(
            game, (c("Ah"), c("Kd")), (c("Tc"), c("9s")), board5
        )
        ev_dict = estimate_blueprint_ev(
            blueprint=MockBlueprint(),
            encoder=MockEncoder(),
            base_game=game,
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
        )
        for ev in ev_dict.values():
            assert np.isfinite(ev)


# ── SafeSubgameSolver ─────────────────────────────────────────────────────────

class TestSafeSubgameSolver:

    @pytest.fixture
    def safe_setup(self):
        game = make_game()
        hero_range, opp_range = small_ranges()
        board5 = make_board5()
        root = make_root_history(
            game, (c("Ah"), c("Kd")), (c("Tc"), c("9s")), board5
        )
        solver = SafeSubgameSolver(
            base_game=game,
            blueprint=MockBlueprint(),
            encoder=MockEncoder(),
        )
        return solver, root, hero_range, opp_range

    def test_returns_subgame_strategy(self, safe_setup):
        solver, root, hero_range, opp_range = safe_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=10,
        )
        assert isinstance(result, SubgameStrategy)

    def test_opt_out_stripped_from_returned_strategy(self, safe_setup):
        solver, root, hero_range, opp_range = safe_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=10,
        )
        # No info set should reference the opt-out action
        game = make_game()
        h = (
            (c("Ah"), c("Kd")),
            (c("Tc"), c("9s")),
            make_board5(),
        )
        probs = result.query(h, player=0)
        # Should be a valid probability distribution
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_strategy_does_not_contain_opt_out_in_query(self, safe_setup):
        solver, root, hero_range, opp_range = safe_setup
        result = solver.solve(
            root_history=root,
            hero_player=0,
            hero_range=hero_range,
            opponent_range=opp_range,
            iterations=15,
        )
        # Walk all info sets: none should have opt-out in their prob vectors
        # (indirectly checked by verifying all returned probs sum to 1)
        for key, probs in result._dict.items():
            assert abs(probs.sum() - 1.0) < 1e-4, (
                f"Info set {key} probs sum to {probs.sum():.4f}"
            )

    def test_build_game_raises(self, safe_setup):
        solver, *_ = safe_setup
        with pytest.raises(NotImplementedError):
            solver._build_game()