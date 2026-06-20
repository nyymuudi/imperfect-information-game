"""Sanity tests for the CFR advisor cache (C2 design + A live MC fallback)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deep_cfr.cfr_cache import (
    ADVISOR_DIMS,
    EV_DIMS,
    PROB_DIMS,
    CFRCache,
    CFRCacheMeta,
    encode_key,
    public_state_key,
)
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE


class TestEncodeKey:

    def test_two_distinct_hands_distinct_keys(self):
        k_aa  = encode_key(0, 0, 0, 0, 0, 7, 0, 7)
        k_72o = encode_key(0, 0, 0, 0, 0, 7, 0, 0)
        assert k_aa != k_72o

    def test_field_clamping(self):
        # Width-3 fields clamp values >7 to 7
        k1 = encode_key(0, 0, 0, 0, 0, 0, 0, 9)   # hand_bucket=9 should clamp
        k2 = encode_key(0, 0, 0, 0, 0, 0, 0, 7)
        assert k1 == k2

    def test_negative_clamps_to_zero(self):
        k1 = encode_key(0, 0, -1, 0, 0, 0, 0, 0)
        k2 = encode_key(0, 0,  0, 0, 0, 0, 0, 0)
        assert k1 == k2


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


class TestPublicStateKey:

    def test_returns_uint64_compatible_int(self, game, encoder):
        h = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        k = public_state_key(h, game, encoder)
        assert isinstance(k, int)
        assert k >= 0

    def test_distinct_hero_hands_distinct_keys(self, game, encoder):
        # Same public state but different hero hole cards → distinct keys
        # (because hand_bucket is in the key).
        h_aa  = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        h_72o = ((24, 1), (48, 49), (0, 5, 10, 15, 20))
        k_aa  = public_state_key(h_aa,  game, encoder)
        k_72o = public_state_key(h_72o, game, encoder)
        assert k_aa != k_72o

    def test_postflop_after_actions_changes_key(self, game, encoder):
        # Same deal, different action history → different street/raises →
        # different key.
        base = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        k_preflop = public_state_key(base, game, encoder)
        h_postflop = game.apply_action(base, "k")  # SB calls preflop
        h_postflop = game.apply_action(h_postflop, "c")  # BB checks
        k_postflop = public_state_key(h_postflop, game, encoder)
        assert k_preflop != k_postflop


# ── In-memory cache + lookup ────────────────────────────────────────────────


def _make_synthetic_cache(keys: list[int]) -> CFRCache:
    n = len(keys)
    probs = np.zeros((n, PROB_DIMS), dtype=np.float32)
    evs   = np.zeros((n, EV_DIMS),   dtype=np.float32)
    # Fill with simple test signal: prob mass on slot 0, EVs increasing.
    for i in range(n):
        probs[i, 0] = 0.7
        probs[i, 2] = 0.3
        evs[i, 0]   = 0.1 * (i + 1)
    return CFRCache(
        keys=np.asarray(keys, dtype=np.uint64),
        probs=probs,
        evs=evs,
        meta=CFRCacheMeta(
            starting_stack=50.0,
            raise_fractions=(0.5,),
            max_actions=4,
            iter_per_spot=10,
            n_spots_requested=n,
            timestamp="test",
        ),
    )


class TestCacheLookup:

    def test_hit_returns_probs_and_evs(self, game, encoder):
        h   = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        key = public_state_key(h, game, encoder)
        cache = _make_synthetic_cache([key])
        out = cache.lookup(h, game, encoder)
        assert out is not None
        probs, evs = out
        assert probs.shape == (PROB_DIMS,)
        assert evs.shape   == (EV_DIMS,)
        np.testing.assert_allclose(probs[0], 0.7, atol=1e-6)
        np.testing.assert_allclose(probs[2], 0.3, atol=1e-6)

    def test_miss_returns_none(self, game, encoder):
        cache = _make_synthetic_cache([12345])  # arbitrary, won't match
        h   = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        assert cache.lookup(h, game, encoder) is None


class TestEncoderAugmentation:

    def test_state_size_grows_by_12(self, game, encoder):
        enc_no  = NLHEEncoder(starting_stack=50.0, equity_sims=20)
        h   = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        key = public_state_key(h, game, enc_no)
        enc_with = NLHEEncoder(
            starting_stack=50.0, equity_sims=20,
            cfr_cache=_make_synthetic_cache([key]),
        )
        assert enc_with.state_size() == enc_no.state_size() + ADVISOR_DIMS

    def test_base_dims_invariant(self, game, encoder):
        enc_no  = NLHEEncoder(starting_stack=50.0, equity_sims=20)
        h   = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        key = public_state_key(h, game, enc_no)
        enc_with = NLHEEncoder(
            starting_stack=50.0, equity_sims=20,
            cfr_cache=_make_synthetic_cache([key]),
        )
        v_no   = enc_no.encode(h, 0)
        v_with = enc_with.encode(h, 0)
        # Base 37 dims should match exactly.
        np.testing.assert_allclose(v_no, v_with[:enc_no.state_size()],
                                   atol=1e-6)

    def test_cache_hit_writes_advisor_dims(self, game, encoder):
        enc_no  = NLHEEncoder(starting_stack=50.0, equity_sims=20)
        h   = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        key = public_state_key(h, game, enc_no)
        enc_with = NLHEEncoder(
            starting_stack=50.0, equity_sims=20,
            cfr_cache=_make_synthetic_cache([key]),
        )
        v = enc_with.encode(h, 0)
        # Advisor section starts at base_size = state_size_no_cache
        base = enc_no.state_size()
        advisor_probs = v[base : base + PROB_DIMS]
        np.testing.assert_allclose(advisor_probs[0], 0.7, atol=1e-6)
        np.testing.assert_allclose(advisor_probs[2], 0.3, atol=1e-6)


class TestLiveMCFallback:

    def test_cache_miss_triggers_live_mc(self, game):
        # Build encoder with cache containing zero matching keys.
        enc_no = NLHEEncoder(starting_stack=50.0, equity_sims=20)
        cache_empty = _make_synthetic_cache([99999999])  # no match
        enc_with = NLHEEncoder(
            starting_stack=50.0, equity_sims=20,
            cfr_cache=cache_empty,
        )
        h = ((48, 49), (24, 1), (0, 5, 10, 15, 20))
        v = enc_with.encode(h, 0)
        base = enc_no.state_size()
        advisor_probs = v[base : base + PROB_DIMS]
        # Live MC produces probs that SUM to ~1 over legal slots.
        # (Not necessarily uniform — MC EVs differ across legal actions.)
        assert advisor_probs.sum() == pytest.approx(1.0, abs=1e-3)
