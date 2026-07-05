"""Exact combo-level river vector CFR: removal identities + consistency.

Cross-validated offline against the deal-based UnsafeSubgameSolver on a
shared 40×40-combo river node: mean |Δv| = 0.007 chips (max 0.05) with a
~700× wall-clock speedup (2 s vs 1399 s). These tests lock the cheap
invariants; the expensive cross-check lives in the commit message.
"""

import numpy as np
import pytest

from src.games.postflop_nlhe import PostflopNLHE
from src.search.nlhe_river_vector import (
    COMBOS, N_COMBOS,
    RiverVectorCFR, bucket_map, bucket_range, bucket_values, disjoint_mass,
)

BOARD = (0, 5, 10, 15, 20)
NODE = ((48, 49), (24, 1), BOARD, "k", "c", "c", "c", "c", "c")


@pytest.fixture
def game():
    return PostflopNLHE(starting_stack=50.0, max_raises_per_street=1,
                        raise_fractions=(0.5,))


def _rand_range(rng, n=120):
    v = np.zeros(N_COMBOS)
    bs = set(BOARD)
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    idx = rng.choice(live, size=n, replace=False)
    v[idx] = rng.dirichlet(np.ones(n))
    return v


class TestRemovalIdentities:

    def test_disjoint_mass_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        x = _rand_range(rng, 30)
        fast = disjoint_mass(x)
        for i in rng.choice(N_COMBOS, size=25, replace=False):
            brute = sum(x[j] for j, c in enumerate(COMBOS)
                        if not (set(c) & set(COMBOS[i])))
            np.testing.assert_allclose(fast[i], brute, atol=1e-12)

    def test_showdown_values_match_bruteforce(self, game):
        rng = np.random.default_rng(1)
        x_opp = _rand_range(rng, 12)
        solver = RiverVectorCFR(game, NODE, x_opp, x_opp, iterations=1)
        # showdown terminal: both check the river
        rep = solver._rep(("c", "c"))
        assert game.is_terminal(rep)
        inv = float(game._parse_state(rep)["invested"][0])
        u = solver._showdown_values(rep, x_opp, traverser=0)

        from src.abstraction.equity import evaluate_7card
        for i in rng.choice(np.where(solver.live_mask)[0], size=15,
                            replace=False):
            hero = COMBOS[i]
            s_h = evaluate_7card(tuple(hero) + BOARD)
            brute = 0.0
            for j in np.where(x_opp > 0)[0]:
                opp = COMBOS[j]
                if set(opp) & set(hero):
                    continue
                s_o = evaluate_7card(tuple(opp) + BOARD)
                brute += x_opp[j] * (inv if s_h > s_o
                                     else (-inv if s_h < s_o else 0.0))
            np.testing.assert_allclose(u[i], brute, atol=1e-9)


class TestSolve:

    def test_zero_sum_and_self_consistency(self, game):
        rng = np.random.default_rng(3)
        x0, x1 = _rand_range(rng), _rand_range(rng)
        s = RiverVectorCFR(game, NODE, x0, x1, iterations=150)
        s.solve()
        v0, v1 = s.root_values()
        p0 = x0 * disjoint_mass(x1)
        p1 = x1 * disjoint_mass(x0)
        e0 = (p0 / p0.sum()) @ v0
        e1 = (p1 / p1.sum()) @ v1
        assert abs(e0 + e1) < 1e-9, f"zero-sum violated: {e0 + e1:.2e}"

        s2 = RiverVectorCFR(game, NODE, x0, x1, iterations=600)
        s2.solve()
        w0, _ = s2.root_values()
        m = x0 > 1e-4
        drift = np.abs((v0 - w0)[m]).mean()
        assert drift < 0.05, f"not converged at 150 iters: drift {drift:.4f}"

    def test_dead_combos_zero(self, game):
        rng = np.random.default_rng(4)
        x0, x1 = _rand_range(rng), _rand_range(rng)
        s = RiverVectorCFR(game, NODE, x0, x1, iterations=30)
        s.solve()
        v0, v1 = s.root_values()
        dead = ~s.live_mask
        assert np.all(v0[dead] == 0.0) and np.all(v1[dead] == 0.0)


class TestBucketInterface:

    def test_bucket_map_partitions_live_combos(self):
        bm = bucket_map(BOARD, 50)
        bs = set(BOARD)
        for i, c in enumerate(COMBOS):
            if bs & set(c):
                assert bm[i] == -1
            else:
                assert 0 <= bm[i] < 50

    def test_bucket_range_normalised(self):
        rng = np.random.default_rng(5)
        x = _rand_range(rng)
        bm = bucket_map(BOARD, 50)
        br = bucket_range(x, bm, 50)
        np.testing.assert_allclose(br.sum(), 1.0, atol=1e-12)

    def test_bucket_values_weighted_mean_bounds(self):
        rng = np.random.default_rng(6)
        x = _rand_range(rng)
        bm = bucket_map(BOARD, 50)
        v = rng.normal(size=N_COMBOS)
        v[bm < 0] = 0.0
        bv = bucket_values(v, x, bm, 50)
        assert bv.shape == (50,)
        assert np.all(bv >= v[bm >= 0].min() - 1e-9)
        assert np.all(bv <= v[bm >= 0].max() + 1e-9)
