"""Unit tests for src/search: PBS range updates + Leduc subgame re-solving."""

import numpy as np
import pytest

from src.games.leduc import LeducHoldem
from src.search.pbs import (
    LEDUC_CARDS,
    CARD_IDX,
    initial_pbs,
    representative_history,
    update_on_action,
    update_on_community,
)
from src.search.resolve import LeducSubgame, resolve


@pytest.fixture
def game():
    return LeducHoldem()


def uniform_fn(game):
    def fn(h, p):
        n = len(game.legal_actions(h))
        return np.ones(n) / n
    return fn


class TestPBSUpdates:

    def test_initial_ranges_uniform(self):
        pbs = initial_pbs()
        for p in range(2):
            np.testing.assert_allclose(pbs.range_array(p), np.full(6, 1 / 6))

    def test_bayes_update_known_strategy(self, game):
        """J always raises, Q/K always check → after observed check the
        posterior must be uniform over Q/K cards only."""
        def sigma(h, p):
            legal = game.legal_actions(h)   # root: ['c', 'r']
            rank = h[p][0]
            probs = np.zeros(len(legal))
            probs[legal.index("r" if rank == "J" else "c")] = 1.0
            return probs

        pbs = update_on_action(initial_pbs(), game, 0, "c", sigma)
        r0 = pbs.range_array(0)
        np.testing.assert_allclose(r0[:2], [0.0, 0.0])       # J1, J2 excluded
        np.testing.assert_allclose(r0[2:], np.full(4, 0.25))  # Q/K uniform
        # Non-acting player's range untouched
        np.testing.assert_allclose(pbs.range_array(1), np.full(6, 1 / 6))

    def test_zero_mass_falls_back_to_uniform(self, game):
        """Observing a 0-probability action resets the range (Bayes undefined)."""
        def never_raise(h, p):
            legal = game.legal_actions(h)
            probs = np.zeros(len(legal))
            probs[legal.index("c")] = 1.0
            return probs

        pbs = update_on_action(initial_pbs(), game, 0, "r", never_raise)
        np.testing.assert_allclose(pbs.range_array(0), np.full(6, 1 / 6))

    def test_community_reveal_card_removal(self, game):
        pbs = update_on_community(initial_pbs(), "Q1")
        for p in range(2):
            r = pbs.range_array(p)
            assert r[CARD_IDX["Q1"]] == 0.0
            np.testing.assert_allclose(r.sum(), 1.0)

    def test_representative_history_infoset_consistency(self, game):
        """Rep-history infoset key must match any concrete deal's key."""
        actions = ("c", "r")
        rep = representative_history("K2", 1, None, actions)
        key_rep = game.info_set_key(rep, 1)
        concrete = ("J1", "K2", "Q2") + actions
        assert game.info_set_key(concrete, 1) == key_rep


class TestLeducSubgame:

    def test_root_deals_complete(self, game):
        sg = LeducSubgame(game, initial_pbs())
        deals = sg.initial_histories()
        assert len(deals) == 120           # 30 pairs × 4 community cards
        np.testing.assert_allclose(sum(w for _, w in deals), 1.0)

    def test_card_removal_in_deals(self, game):
        pbs = update_on_community(initial_pbs(), "K1")
        sg = LeducSubgame(game, pbs)
        for h, w in sg.initial_histories():
            assert h[2] == "K1"
            assert "K1" not in (h[0], h[1])
            assert h[0] != h[1]

    def test_prefix_applied(self, game):
        pbs = update_on_action(initial_pbs(), game, 0, "c", uniform_fn(game))
        sg = LeducSubgame(game, pbs)
        h, _ = sg.initial_histories()[0]
        assert h[3:] == ("c",)
        assert sg.current_player(h) == 1


class TestResolve:

    def test_resolve_root_approaches_equilibrium(self, game):
        """CFR+ on the full-game 'subgame' must reach low exploitability —
        validates that the subgame construction preserves the game."""
        res = resolve(game, initial_pbs(), iterations=200)
        # Load into a fresh solver carrier for the exact BR check.
        from src.solvers.cfr import CFRSolver
        ref = CFRSolver(game=game, linear_averaging=True)
        for key, data in res.solver.info_sets.items():
            carried = ref._get_or_create_info_set(key, data.actions)
            carried.cumulative_strategy = data.average_strategy().copy()
        expl = ref.exploitability()
        # Tabular CFR+ @200 iters on Leduc lands well under 0.5.
        assert expl < 0.5, f"root resolve exploitability {expl:.3f} too high"

    def test_strategies_normalised(self, game):
        res = resolve(game, initial_pbs(), iterations=20)
        for key, s in res.strategies.items():
            np.testing.assert_allclose(s.sum(), 1.0, atol=1e-9)
