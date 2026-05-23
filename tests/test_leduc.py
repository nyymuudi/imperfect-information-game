"""
Test suite for Leduc Hold'em.

Tests:
    1. Game tree structure (info sets, terminals, initial states)
    2. Game mechanics (terminal detection, payoffs, player turns)
    3. Info set correctness (hides opponent card, reveals community after R1)
    4. CFR convergence (basic sanity on strategy structure)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.games.leduc import LeducHoldem


@pytest.fixture
def game():
    return LeducHoldem()


# ── Game Tree Structure ─────────────────────────────────────────


class TestLeducTree:
    """Verify tree shape matches known Leduc properties."""

    def setup_method(self):
        self.game = LeducHoldem()
        self.info_sets = set()
        self.terminals = 0

        def traverse(hist):
            if self.game.is_terminal(hist):
                self.terminals += 1
                return
            player = self.game.current_player(hist)
            info = self.game.info_set_key(hist, player)
            self.info_sets.add(info)
            for a in self.game.legal_actions(hist):
                traverse(self.game.apply_action(hist, a))

        for h, _ in self.game.initial_histories():
            traverse(h)

    def test_initial_histories_count(self):
        """30 dealings × 4 community cards = 120."""
        assert len(self.game.initial_histories()) == 120

    def test_initial_probs_sum_to_one(self):
        total = sum(p for _, p in self.game.initial_histories())
        assert abs(total - 1.0) < 1e-10

    def test_info_set_count(self):
        """Leduc with max 2 raises/round should have 288 info sets."""
        assert len(self.info_sets) == 288

    def test_terminal_count_positive(self):
        assert self.terminals > 0

    def test_no_infinite_recursion(self):
        """Tree traversal terminates (implicitly tested by setup)."""
        assert self.terminals > 100


# ── Game Mechanics ──────────────────────────────────────────────


class TestLeducMechanics:

    def setup_method(self):
        self.game = LeducHoldem()
        # Standard starting history: J1 vs Q1, community K1
        self.h = ("J1", "Q1", "K1")

    def test_num_players(self):
        assert self.game.num_players() == 2

    def test_not_terminal_at_start(self):
        assert not self.game.is_terminal(self.h)

    def test_p0_acts_first(self):
        assert self.game.current_player(self.h) == 0

    def test_legal_actions_at_start(self):
        actions = self.game.legal_actions(self.h)
        assert "c" in actions
        assert "r" in actions
        assert "f" not in actions  # Nothing to fold to

    # ── Round 1 terminals ──

    def test_fold_to_raise_terminal(self):
        h = self.h + ("r", "f")
        assert self.game.is_terminal(h)

    def test_check_check_not_terminal_leduc(self):
        """In Leduc, check-check ends round 1 but game continues to round 2."""
        h = self.h + ("c", "c")
        assert not self.game.is_terminal(h)

    def test_raise_call_not_terminal(self):
        """Raise-call ends round 1, not the game."""
        h = self.h + ("r", "k")
        assert not self.game.is_terminal(h)

    # ── Round 2 ──

    def test_round_2_starts_after_check_check(self):
        h = self.h + ("c", "c")
        assert not self.game.is_terminal(h)
        assert self.game.current_player(h) == 0  # P0 acts first in R2

    def test_round_2_check_check_terminal(self):
        h = self.h + ("c", "c", "c", "c")
        assert self.game.is_terminal(h)

    def test_round_2_raise_call_terminal(self):
        h = self.h + ("c", "c", "r", "k")
        assert self.game.is_terminal(h)

    def test_round_2_raise_fold_terminal(self):
        h = self.h + ("c", "c", "r", "f")
        assert self.game.is_terminal(h)

    def test_round_2_after_raise_call(self):
        h = self.h + ("r", "k")
        assert not self.game.is_terminal(h)
        p = self.game.current_player(h)
        assert p == 0  # P0 acts first in R2

    # ── Payoffs ──

    def test_fold_payoff_r1(self):
        """P0 raises, P1 folds → P0 wins P1's ante (1)."""
        h = self.h + ("r", "f")
        payoffs = self.game.terminal_payoffs(h)
        assert payoffs[0] > 0
        assert payoffs[1] < 0

    def test_showdown_higher_card_wins(self):
        """J vs Q, community K → Q wins (higher unpaired card)."""
        h = ("J1", "Q1", "K1", "c", "c", "c", "c")
        payoffs = self.game.terminal_payoffs(h)
        assert payoffs[0] < 0  # J loses
        assert payoffs[1] > 0  # Q wins

    def test_showdown_pair_beats_high(self):
        """J vs Q, community J → J wins (pair)."""
        h = ("J1", "Q1", "J2", "c", "c", "c", "c")
        payoffs = self.game.terminal_payoffs(h)
        assert payoffs[0] > 0  # J has pair, wins
        assert payoffs[1] < 0

    def test_payoff_zero_sum(self):
        """All payoffs must be zero-sum."""
        terminals = [
            ("J1", "Q1", "K1", "c", "c", "c", "c"),
            ("J1", "Q1", "K1", "r", "f"),
            ("J1", "Q1", "K1", "r", "k", "r", "k"),
            ("J1", "Q1", "J2", "c", "c", "c", "c"),
            ("K1", "J1", "Q1", "r", "k", "c", "c"),
        ]
        for h in terminals:
            p = self.game.terminal_payoffs(h)
            assert abs(p[0] + p[1]) < 1e-10, f"Non-zero-sum: {h} → {p}"

    def test_raise_increases_pot(self):
        """Showdown after raises should have larger payoffs than check-check."""
        h_cc = ("K1", "J1", "Q1", "c", "c", "c", "c")
        h_rk = ("K1", "J1", "Q1", "r", "k", "c", "c")
        p_cc = self.game.terminal_payoffs(h_cc)
        p_rk = self.game.terminal_payoffs(h_rk)
        assert abs(p_rk[0]) > abs(p_cc[0])


# ── Info Sets ───────────────────────────────────────────────────


class TestLeducInfoSets:

    def setup_method(self):
        self.game = LeducHoldem()

    def test_hides_opponent_card(self):
        k1 = self.game.info_set_key(("J1", "Q1", "K1"), 0)
        k2 = self.game.info_set_key(("J1", "K1", "Q1"), 0)
        assert k1 == k2  # P0 has J1, doesn't see opponent

    def test_community_hidden_in_round_1(self):
        """Community card not visible during round 1."""
        k1 = self.game.info_set_key(("J1", "Q1", "K1"), 0)
        k2 = self.game.info_set_key(("J1", "Q1", "Q2"), 0)
        assert k1 == k2  # Different community, same info set in R1

    def test_community_visible_in_round_2(self):
        """Community card visible after round 1 completes."""
        h1 = ("J1", "Q1", "K1", "c", "c")
        h2 = ("J1", "Q1", "Q2", "c", "c")
        k1 = self.game.info_set_key(h1, 0)
        k2 = self.game.info_set_key(h2, 0)
        assert k1 != k2  # Different community → different info set in R2

    def test_same_rank_same_info_set(self):
        """J1 and J2 are indistinguishable to the player."""
        k1 = self.game.info_set_key(("J1", "Q1", "K1"), 0)
        k2 = self.game.info_set_key(("J2", "Q1", "K1"), 0)
        assert k1 == k2

    def test_actions_in_info_set(self):
        h = ("J1", "Q1", "K1", "r")
        k = self.game.info_set_key(h, 1)
        assert "r" in k


# ── CFR Sanity ──────────────────────────────────────────────────


class TestLeducCFR:
    """Quick CFR run to verify solver works on Leduc."""

    @pytest.fixture(autouse=True)
    def solve(self):
        from src.solvers.cfr import CFRSolver
        self.game = LeducHoldem()
        self.solver = CFRSolver(game=self.game, linear_averaging=True)
        self.strategy = self.solver.solve(iterations=100)

    def test_discovers_all_info_sets(self):
        assert len(self.strategy) == 288

    def test_strategies_are_valid(self):
        for key, strat in self.strategy.items():
            assert all(s >= -1e-10 for s in strat), f"Negative at {key}"
            assert abs(strat.sum() - 1.0) < 1e-6, f"Sum≠1 at {key}"

    def test_king_raises_more_than_jack_r1(self):
        """In round 1, K should raise more than J (on average)."""
        j_raise = 0
        k_raise = 0
        count_j = count_k = 0
        for key, strat in self.strategy.items():
            parts = key.split("|")
            rank, comm, actions = parts[0], parts[1], parts[2]
            if comm == "" and actions == "":
                # Round 1 opening
                r_idx = self.game.legal_actions(
                    ("J1", "Q1", "K1")
                ).index("r")
                if rank == "J":
                    j_raise += strat[r_idx]
                    count_j += 1
                elif rank == "K":
                    k_raise += strat[r_idx]
                    count_k += 1
        if count_j > 0 and count_k > 0:
            assert k_raise / count_k > j_raise / count_j


if __name__ == "__main__":
    pytest.main([__file__, "-v"])