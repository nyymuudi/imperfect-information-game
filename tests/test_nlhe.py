"""
Test suite for Preflop NLHE + card abstraction.

Uses a minimal 3-bucket abstraction with low-sim equity for speed.
Tests structured in three tiers:
    1. Equity & abstraction correctness
    2. Game mechanics (terminal states, payoffs, legal actions)
    3. CFR convergence (strategy structure validates GTO properties)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.abstraction.equity import (
    evaluate_5card, evaluate_7card,
    str_to_card, canonical_hand_class,
    all_169_classes, representative_hand, equity_vs_random,
)
from src.abstraction.card_abstraction import CardAbstraction
from src.games.nlhe_preflop import PreflopNLHE
from src.solvers.cfr import CFRSolver


# ── Tier 1: Equity & Abstraction ────────────────────────────────


class TestHandEvaluator:
    """Verify poker hand evaluation correctness."""

    def test_pair_beats_high_card(self):
        pair = (str_to_card("Ac"), str_to_card("Ad"),
                str_to_card("Kh"), str_to_card("7s"), str_to_card("3d"))
        high = (str_to_card("Kc"), str_to_card("Qd"),
                str_to_card("Jh"), str_to_card("7s"), str_to_card("3d"))
        assert evaluate_5card(pair) > evaluate_5card(high)

    def test_two_pair_beats_pair(self):
        two_pair = (str_to_card("Ac"), str_to_card("Ad"),
                    str_to_card("Kh"), str_to_card("Ks"), str_to_card("3d"))
        pair = (str_to_card("Ac"), str_to_card("Ad"),
                str_to_card("Kh"), str_to_card("Qs"), str_to_card("3d"))
        assert evaluate_5card(two_pair) > evaluate_5card(pair)

    def test_flush_beats_straight(self):
        flush = (str_to_card("Ah"), str_to_card("Th"), 
                 str_to_card("7h"), str_to_card("4h"), str_to_card("2h"))
        straight = (str_to_card("9c"), str_to_card("8d"),
                    str_to_card("7h"), str_to_card("6s"), str_to_card("5c"))
        assert evaluate_5card(flush) > evaluate_5card(straight)

    def test_full_house_beats_flush(self):
        full_house = (str_to_card("Ac"), str_to_card("Ad"),
                      str_to_card("Ah"), str_to_card("Ks"), str_to_card("Kd"))
        flush = (str_to_card("Ah"), str_to_card("Th"),
                 str_to_card("7h"), str_to_card("4h"), str_to_card("2h"))
        assert evaluate_5card(full_house) > evaluate_5card(flush)

    def test_quads_beats_full_house(self):
        quads = (str_to_card("Ac"), str_to_card("Ad"),
                 str_to_card("Ah"), str_to_card("As"), str_to_card("Kd"))
        full_house = (str_to_card("Kc"), str_to_card("Kd"),
                      str_to_card("Kh"), str_to_card("Qs"), str_to_card("Qd"))
        assert evaluate_5card(quads) > evaluate_5card(full_house)

    def test_straight_flush_beats_quads(self):
        sf = (str_to_card("9h"), str_to_card("8h"),
              str_to_card("7h"), str_to_card("6h"), str_to_card("5h"))
        quads = (str_to_card("Ac"), str_to_card("Ad"),
                 str_to_card("Ah"), str_to_card("As"), str_to_card("Kd"))
        assert evaluate_5card(sf) > evaluate_5card(quads)

    def test_wheel_straight(self):
        """A-2-3-4-5 is a valid straight."""
        wheel = (str_to_card("Ac"), str_to_card("2d"),
                 str_to_card("3h"), str_to_card("4s"), str_to_card("5c"))
        high_card = (str_to_card("Ac"), str_to_card("Kd"),
                     str_to_card("8h"), str_to_card("4s"), str_to_card("2c"))
        assert evaluate_5card(wheel) > evaluate_5card(high_card)

    def test_higher_pair_wins(self):
        aa = (str_to_card("Ac"), str_to_card("Ad"),
              str_to_card("7h"), str_to_card("4s"), str_to_card("2c"))
        kk = (str_to_card("Kc"), str_to_card("Kd"),
              str_to_card("7h"), str_to_card("4s"), str_to_card("2c"))
        assert evaluate_5card(aa) > evaluate_5card(kk)

    def test_7card_finds_best(self):
        """7-card evaluator should find the flush in 7 cards."""
        cards = tuple(str_to_card(c) for c in 
                      ["Ah", "Kh", "Qh", "Jh", "3h", "2c", "4d"])
        val = evaluate_7card(cards)
        # Should be at least a flush (category 5)
        pure_flush = tuple(str_to_card(c) for c in
                          ["Ah", "Kh", "Qh", "Jh", "3h"])
        assert val >= evaluate_5card(pure_flush)


class TestEquity:
    """Verify equity calculations are in correct ranges."""

    def test_aa_equity_above_80(self):
        cards = representative_hand("AA")
        eq = equity_vs_random(cards, num_simulations=5000,
                              rng=np.random.default_rng(42))
        assert eq > 0.80

    def test_72o_equity_below_40(self):
        cards = representative_hand("72o")
        eq = equity_vs_random(cards, num_simulations=5000,
                              rng=np.random.default_rng(42))
        assert eq < 0.40

    def test_aa_beats_72o_in_equity(self):
        aa = equity_vs_random(representative_hand("AA"), 3000,
                              rng=np.random.default_rng(1))
        bad = equity_vs_random(representative_hand("72o"), 3000,
                               rng=np.random.default_rng(1))
        assert aa > bad

    def test_suited_beats_offsuit(self):
        aks = equity_vs_random(representative_hand("AKs"), 5000,
                               rng=np.random.default_rng(42))
        ako = equity_vs_random(representative_hand("AKo"), 5000,
                               rng=np.random.default_rng(42))
        assert aks > ako


class TestCanonicalHands:
    """Verify hand classification."""

    def test_169_classes(self):
        assert len(all_169_classes()) == 169

    def test_pair_classification(self):
        c1, c2 = str_to_card("Ac"), str_to_card("Ad")
        assert canonical_hand_class(c1, c2) == "AA"

    def test_suited_classification(self):
        c1, c2 = str_to_card("Ah"), str_to_card("Kh")
        assert canonical_hand_class(c1, c2) == "AKs"

    def test_offsuit_classification(self):
        c1, c2 = str_to_card("Ah"), str_to_card("Kd")
        assert canonical_hand_class(c1, c2) == "AKo"

    def test_order_invariant(self):
        c1, c2 = str_to_card("7d"), str_to_card("Ac")
        assert canonical_hand_class(c1, c2) == "A7o"
        assert canonical_hand_class(c2, c1) == "A7o"


class TestCardAbstraction:
    """Verify abstraction bucketing."""

    @pytest.fixture(autouse=True)
    def build_abstraction(self):
        self.abstraction = CardAbstraction.from_equity(
            num_buckets=4, num_simulations=200, seed=42
        )

    def test_correct_bucket_count(self):
        assert self.abstraction.num_buckets == 4

    def test_all_hands_assigned(self):
        assert len(self.abstraction.hand_to_bucket) == 169

    def test_buckets_in_range(self):
        for bucket in self.abstraction.hand_to_bucket.values():
            assert 0 <= bucket < 4

    def test_aa_in_highest_bucket(self):
        assert self.abstraction.get_bucket("AA") == 3

    def test_72o_in_lowest_bucket(self):
        b = self.abstraction.get_bucket("72o")
        assert b <= 1  # Should be in bottom half


# ── Tier 2: Game Mechanics ──────────────────────────────────────


class TestNLHEGameMechanics:
    """Verify NLHE game rules."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.abstraction = CardAbstraction.from_equity(
            num_buckets=3, num_simulations=100, seed=42
        )
        self.game = PreflopNLHE(
            abstraction=self.abstraction,
            stack_bb=20.0,
            raise_sizes=(3.0, 8.0),
        )

    def test_num_players(self):
        assert self.game.num_players() == 2

    def test_initial_histories_count(self):
        # 3 buckets × 3 buckets = 9 pairs
        assert len(self.game.initial_histories()) == 9

    def test_initial_probs_sum_to_one(self):
        total = sum(p for _, p in self.game.initial_histories())
        assert abs(total - 1.0) < 1e-6

    def test_sb_acts_first(self):
        h = (1, 2)  # bucket pair
        assert self.game.current_player(h) == 0

    def test_fold_is_terminal(self):
        assert self.game.is_terminal((1, 2, "f"))

    def test_call_call_is_terminal(self):
        # SB limps, BB checks
        assert self.game.is_terminal((1, 2, "c", "c"))

    def test_raise_call_is_terminal(self):
        assert self.game.is_terminal((1, 2, "r", "c"))

    def test_raise_fold_is_terminal(self):
        assert self.game.is_terminal((1, 2, "r", "f"))

    def test_single_raise_not_terminal(self):
        assert not self.game.is_terminal((1, 2, "r"))

    def test_fold_payoff_sb_loses(self):
        # SB folds, loses 0.5BB ante
        payoffs = self.game.terminal_payoffs((1, 2, "f"))
        assert payoffs[0] < 0  # SB loses
        assert payoffs[1] > 0  # BB wins

    def test_fold_payoff_bb_loses(self):
        # SB raises, BB folds, BB loses 1BB ante
        payoffs = self.game.terminal_payoffs((1, 2, "r", "f"))
        assert payoffs[0] > 0  # SB wins
        assert payoffs[1] < 0  # BB loses

    def test_info_set_hides_opponent(self):
        k1 = self.game.info_set_key((1, 0), 0)
        k2 = self.game.info_set_key((1, 2), 0)
        assert k1 == k2  # P0 sees bucket 1, doesn't know opponent

    def test_info_set_includes_actions(self):
        k = self.game.info_set_key((1, 2, "r"), 1)
        assert "r" in k

    def test_legal_actions_at_root(self):
        actions = self.game.legal_actions((1, 2))
        assert "f" in actions  # Can fold (facing BB)
        assert "c" in actions  # Can call/limp

    def test_no_fold_when_checked_to(self):
        # BB after SB limps — no bet to fold to
        actions = self.game.legal_actions((1, 2, "c"))
        assert "f" not in actions


# ── Tier 3: CFR Convergence ─────────────────────────────────────


class TestNLHECFR:
    """Verify CFR produces sensible strategies on abstracted NLHE."""

    @pytest.fixture(autouse=True)
    def solve(self):
        self.abstraction = CardAbstraction.from_equity(
            num_buckets=3, num_simulations=100, seed=42
        )
        self.game = PreflopNLHE(
            abstraction=self.abstraction,
            stack_bb=20.0,
            raise_sizes=(3.0, 8.0),
        )
        self.solver = CFRSolver(game=self.game, linear_averaging=True)
        self.strategy = self.solver.solve(iterations=500)

    def test_info_sets_discovered(self):
        assert len(self.strategy) > 0

    def test_strategies_valid_distributions(self):
        for key, strat in self.strategy.items():
            assert all(s >= -1e-10 for s in strat), f"Negative at {key}"
            assert abs(strat.sum() - 1.0) < 1e-6, f"Sum≠1 at {key}"

    def test_strongest_bucket_rarely_folds(self):
        """Strongest bucket should almost never fold as SB opener."""
        max_bucket = self.abstraction.num_buckets - 1
        key = f"B{max_bucket}|"
        if key in self.strategy:
            actions = self.game.legal_actions((max_bucket, 0))
            if "f" in actions:
                fold_idx = actions.index("f")
                assert self.strategy[key][fold_idx] < 0.1

    def test_weakest_bucket_folds_most(self):
        """Weakest bucket should fold more than strongest."""
        weak_key = "B0|"
        strong_key = f"B{self.abstraction.num_buckets - 1}|"
        if weak_key in self.strategy and strong_key in self.strategy:
            w_actions = self.game.legal_actions((0, 0))
            s_actions = self.game.legal_actions(
                (self.abstraction.num_buckets - 1, 0)
            )
            if "f" in w_actions and "f" in s_actions:
                w_fold = self.strategy[weak_key][w_actions.index("f")]
                s_fold = self.strategy[strong_key][s_actions.index("f")]
                assert w_fold > s_fold

    def test_bb_folds_weak_to_raise(self):
        """BB with weakest bucket should usually fold to a raise."""
        key = "B0|r"
        if key in self.strategy:
            actions = self.game.legal_actions((0, 0, "r"))
            if "f" in actions:
                fold_idx = actions.index("f")
                assert self.strategy[key][fold_idx] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])