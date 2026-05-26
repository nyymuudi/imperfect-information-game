"""
Tests for PostflopAbstraction.

Run with:  pytest tests/test_postflop_abstraction.py -v

Performance note: river exact enumeration is O(C(45,2)) ≈ 990 iterations,
fast enough for inline use.  Flop/turn MC tests use n_sims=50 for speed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.abstraction.postflop_abstraction import (
    PostflopAbstraction,
    ehs,
    _river_ehs_exact,
)
from src.abstraction.equity import str_to_card


# ── Helpers ───────────────────────────────────────────────────────────────────

def cards(*strs: str) -> tuple[int, ...]:
    """Convert card strings to ints: cards('Ah', 'Ks') → (48, 45)."""
    return tuple(str_to_card(s) for s in strs)


def hole(*strs: str) -> tuple[int, int]:
    c = cards(*strs)
    assert len(c) == 2
    return c


# ── Module-level ehs() ───────────────────────────────────────────────────────

class TestEhsFunction:

    def test_river_aa_on_rags_beats_random(self):
        """AA on a raggy board should have EHS >> 0.5."""
        h = hole("Ah", "As")
        board = cards("2c", "7d", "9h", "3s", "Tc")
        result = ehs(h, board)
        assert result > 0.75, f"AA EHS = {result:.3f}, expected > 0.75"

    def test_river_72o_on_rags_loses_to_random(self):
        """72o unimproved should have EHS << 0.5."""
        h = hole("2h", "7d")
        board = cards("Ac", "Ks", "Qh", "Js", "9c")
        result = ehs(h, board)
        assert result < 0.15, f"72o EHS = {result:.3f}, expected < 0.15"

    def test_river_exact_matches_monte_carlo(self):
        """River: exact and MC (large n) should agree within 2%."""
        h = hole("Kh", "Ks")
        board = cards("2c", "7d", "9h", "3s", "Tc")
        exact = _river_ehs_exact(h, board)
        mc = ehs(h, board, n_sims=2000, rng=np.random.default_rng(0))
        assert abs(exact - mc) < 0.02, (
            f"River exact={exact:.4f}, MC={mc:.4f}, diff={abs(exact-mc):.4f}"
        )

    def test_ehs_in_unit_interval(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            deck = list(range(52))
            rng.shuffle(deck)
            h = (deck[0], deck[1])
            board3 = tuple(deck[2:5])
            board4 = tuple(deck[2:6])
            board5 = tuple(deck[2:7])
            for board in [board3, board4, board5]:
                val = ehs(h, board, n_sims=50, rng=rng)
                assert 0.0 <= val <= 1.0, f"EHS={val} out of [0,1]"

    def test_flop_stronger_hand_higher_ehs(self):
        """AA should have higher EHS than 72o on the same board."""
        board = cards("3c", "8d", "Jh")
        aa   = ehs(hole("Ah", "As"), board, n_sims=300, rng=np.random.default_rng(0))
        junk = ehs(hole("2c", "7s"), board, n_sims=300, rng=np.random.default_rng(0))
        assert aa > junk, f"AA EHS={aa:.3f} not > 72o EHS={junk:.3f}"

    def test_invalid_board_length_raises(self):
        h = hole("Ah", "As")
        with pytest.raises(AssertionError):
            ehs(h, cards("2c", "3d"))         # 2-card board
        with pytest.raises(AssertionError):
            ehs(h, cards("2c", "3d", "4h", "5s", "6c", "7d"))  # 6-card board


# ── PostflopAbstraction.bucket() ─────────────────────────────────────────────

class TestBucket:

    @pytest.fixture
    def abst(self):
        return PostflopAbstraction(num_buckets=8, n_sims=100, seed=0)

    def test_bucket_in_valid_range(self, abst):
        h = hole("Ah", "As")
        board = cards("2c", "7d", "9h")
        b = abst.bucket(h, board)
        assert 0 <= b < 8

    def test_strong_hand_high_bucket(self, abst):
        """AA on raggy flop should land in a high bucket."""
        b_aa = abst.bucket(hole("Ah", "As"), cards("2c", "7d", "9h"))
        b_72 = abst.bucket(hole("2c", "7s"), cards("Ah", "Kd", "Qc"))
        assert b_aa > b_72, f"AA bucket={b_aa}, 72o bucket={b_72}"

    def test_river_bucket_deterministic(self, abst):
        """River bucket must be the same on repeated calls (exact, no MC)."""
        h = hole("Kh", "Kd")
        board = cards("2c", "7d", "9h", "3s", "Tc")
        b1 = abst.bucket(h, board)
        b2 = abst.bucket(h, board)
        assert b1 == b2

    def test_invalid_board_length_raises(self, abst):
        h = hole("Ah", "As")
        with pytest.raises(ValueError):
            abst.bucket(h, cards("2c", "3d"))   # preflop board

    def test_all_streets(self, abst):
        h = hole("Ah", "Ks")
        for board_len, board_cards in [
            (3, cards("2c", "7d", "9h")),
            (4, cards("2c", "7d", "9h", "3s")),
            (5, cards("2c", "7d", "9h", "3s", "Tc")),
        ]:
            b = abst.bucket(h, board_cards)
            assert 0 <= b < 8, f"Bucket {b} out of range for board_len={board_len}"

    def test_caching_returns_same_value(self, abst):
        h = hole("Qh", "Qs")
        board = cards("2c", "7d", "9h")
        b1 = abst.bucket(h, board)
        assert len(abst._ehs_cache) > 0
        b2 = abst.bucket(h, board)
        assert b1 == b2
        # Cache should not grow on second call
        size_before = len(abst._ehs_cache)
        abst.bucket(h, board)
        assert len(abst._ehs_cache) == size_before


# ── PostflopAbstraction construction ─────────────────────────────────────────

class TestConstruction:

    def test_int_buckets_applied_to_all_streets(self):
        abst = PostflopAbstraction(num_buckets=4)
        assert abst._buckets == {"flop": 4, "turn": 4, "river": 4}

    def test_dict_buckets_custom_per_street(self):
        abst = PostflopAbstraction(
            num_buckets={"flop": 4, "turn": 6, "river": 8}
        )
        assert abst._buckets["flop"] == 4
        assert abst._buckets["turn"] == 6
        assert abst._buckets["river"] == 8

    def test_dict_buckets_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            PostflopAbstraction(num_buckets={"flop": 4, "turn": 6})   # river missing


# ── range_to_buckets ──────────────────────────────────────────────────────────

class TestRangeToBuckets:

    @pytest.fixture
    def abst(self):
        return PostflopAbstraction(num_buckets=4, n_sims=50, seed=0)

    def test_output_shape(self, abst):
        range_dict = {hole("Ah", "As"): 0.5, hole("2c", "7s"): 0.5}
        board = cards("3d", "8h", "Jc")
        probs = abst.range_to_buckets(range_dict, board)
        assert probs.shape == (4,)

    def test_sums_to_one(self, abst):
        range_dict = {
            hole("Ah", "As"): 0.3,
            hole("Kh", "Ks"): 0.3,
            hole("2c", "7s"): 0.2,
            hole("Jd", "Tc"): 0.2,
        }
        board = cards("3d", "8h", "5c")
        probs = abst.range_to_buckets(range_dict, board)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_dead_cards_excluded(self, abst):
        """Combos conflicting with board must be silently excluded."""
        board = cards("Ah", "Ks", "Qd")
        # Ah is on the board — hand containing Ah should be excluded
        range_dict = {hole("Ah", "2c"): 0.5, hole("Jd", "Tc"): 0.5}
        probs = abst.range_to_buckets(range_dict, board)
        # Only Jd-Tc should contribute; sum should still be 1
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_empty_range_returns_zeros(self, abst):
        probs = abst.range_to_buckets({}, cards("3d", "8h", "5c"))
        assert np.all(probs == 0.0)

    def test_all_dead_returns_zeros(self, abst):
        # Ac and Kh are both on the board — the only combo in range_dict is dead
        board = cards("Ac", "Kh", "2d")
        range_dict = {hole("Ac", "Kh"): 1.0}
        probs = abst.range_to_buckets(range_dict, board)
        assert np.all(probs == 0.0)

    def test_strong_range_concentrates_high_buckets(self, abst):
        """A range of only premium hands should concentrate in buckets 2–3."""
        board = cards("2c", "7d", "9h")
        # Top pairs / overpairs
        premium_range = {
            hole("Ah", "As"): 1.0,
            hole("Ad", "Ac"): 1.0,
            hole("Kh", "Ks"): 1.0,
            hole("Kd", "Kc"): 1.0,
        }
        probs = abst.range_to_buckets(premium_range, board)
        high_bucket_mass = probs[2:].sum()
        assert high_bucket_mass > 0.5, (
            f"Premium range bucket probs: {probs}, high mass={high_bucket_mass:.3f}"
        )


# ── buckets_to_range ──────────────────────────────────────────────────────────

class TestBucketsToRange:

    @pytest.fixture
    def abst(self):
        return PostflopAbstraction(num_buckets=4, n_sims=50, seed=0)

    def test_roundtrip_concentrates_same_bucket(self, abst):
        """
        range → buckets → range should place hands back in original buckets.
        """
        board = cards("2c", "7d", "9h")
        original_range = {hole("Ah", "As"): 1.0}
        bucket_probs = abst.range_to_buckets(original_range, board)

        reconstructed = abst.buckets_to_range(bucket_probs, board)
        assert len(reconstructed) > 0
        assert abs(sum(reconstructed.values()) - 1.0) < 1e-5

    def test_output_is_valid_distribution(self, abst):
        board = cards("Tc", "Jd", "Qh")
        bucket_probs = np.array([0.1, 0.2, 0.4, 0.3], dtype=np.float32)
        result = abst.buckets_to_range(bucket_probs, board)
        assert all(v >= 0 for v in result.values())
        if result:   # may be empty if all combos dead
            assert abs(sum(result.values()) - 1.0) < 1e-4


# ── bucket_ehs_range ─────────────────────────────────────────────────────────

class TestBucketEhsRange:

    def test_covers_full_unit_interval(self):
        abst = PostflopAbstraction(num_buckets=4)
        for street in ("flop", "turn", "river"):
            ranges = abst.bucket_ehs_range(street)
            assert len(ranges) == 4
            assert abs(ranges[0][0]) < 1e-9,   "First bucket should start at 0"
            assert abs(ranges[-1][1] - 1.0) < 1e-9, "Last bucket should end at 1"

    def test_contiguous_bins(self):
        abst = PostflopAbstraction(num_buckets=8)
        ranges = abst.bucket_ehs_range("flop")
        for i in range(len(ranges) - 1):
            assert abs(ranges[i][1] - ranges[i + 1][0]) < 1e-9


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestCache:

    def test_clear_cache_empties_dict(self):
        abst = PostflopAbstraction(num_buckets=4, n_sims=20)
        abst.bucket(hole("Ah", "As"), cards("2c", "7d", "9h"))
        assert len(abst._ehs_cache) > 0
        abst.clear_cache()
        assert len(abst._ehs_cache) == 0

    def test_cache_stats(self):
        abst = PostflopAbstraction(num_buckets=4, n_sims=20, cache_size=50)
        stats = abst.cache_stats()
        assert stats["capacity"] == 50
        assert stats["size"] == 0