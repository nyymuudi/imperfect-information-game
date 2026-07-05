"""Vector CFR: expand-mode equivalence + depth-limit leaf conversion."""

import numpy as np
import pytest

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.search.pbs import LEDUC_CARDS, CARD_IDX, initial_pbs
from src.search.vector_cfr import VectorCFR
from src.search.cfv import (
    exact_leaf_evaluator,
    exact_round2_cfvs,
    pbs_for_round2,
    _ev_under_strategy,
)
from src.search.resolve import resolve


@pytest.fixture
def game():
    return LeducHoldem()


class TestExpandMode:

    def test_full_game_reaches_low_exploitability(self, game):
        v = VectorCFR(game, initial_pbs(), leaf_evaluator=None, iterations=200)
        v.solve()
        strategies = v.average_strategies()

        ref = CFRSolver(game=game, linear_averaging=True)

        def walk(h):
            if game.is_terminal(h):
                return
            p = game.current_player(h)
            acts = game.legal_actions(h)
            key = game.info_set_key(h, p)
            if key not in ref.info_sets:
                probs = strategies.get(key, np.ones(len(acts)) / len(acts))
                d = ref._get_or_create_info_set(key, acts)
                d.cumulative_strategy = np.asarray(
                    probs[: len(acts)], dtype=np.float64
                ).copy()
            for a in acts:
                walk(game.apply_action(h, a))

        for ih, _ in game.initial_histories():
            walk(ih)
        expl = ref.exploitability()
        assert expl < 0.5, f"vector CFR expand-mode exploitability {expl:.3f}"

    def test_strategies_normalised(self, game):
        v = VectorCFR(game, initial_pbs(), iterations=10)
        v.solve()
        for key, probs in v.average_strategies().items():
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-9)


class TestLeafConversion:
    """The evaluate-mode reveal formula must equal direct enumeration.

    u_tr(h) = Σ_c 1/4 · Σ_{h'∉{h,c}} x_opp(h') · EV_tr(h, h', c | σ*_c)

    where σ*_c is the solved round-2 strategy at (c, cont, masked ranges) —
    the exact quantity the oracle evaluator encapsulates as v(h) with the
    (S_c − x_opp(h)) unnormalisation applied by _reveal_boundary.
    """

    @pytest.mark.parametrize("traverser", [0, 1])
    def test_reveal_boundary_matches_bruteforce(self, game, traverser):
        actions = ("r", "k")           # round 1 complete, cont = 3
        cont = 3.0
        rng = np.random.default_rng(7)
        x_tr = rng.random(6)
        x_opp = rng.random(6)
        solve_iters = 60

        v = VectorCFR(game, initial_pbs(),
                      leaf_evaluator=exact_leaf_evaluator(game, solve_iters),
                      iterations=1)
        u_formula = v._reveal_boundary(actions, x_tr.copy(), x_opp.copy(),
                                       traverser)

        x0 = x_tr if traverser == 0 else x_opp
        x1 = x_opp if traverser == 0 else x_tr
        u_direct = np.zeros(6)
        for c in LEDUC_CARDS:
            ci = CARD_IDX[c]
            m0 = x0.copy(); m0[ci] = 0.0
            m1 = x1.copy(); m1[ci] = 0.0
            y0, y1 = m0 / m0.sum(), m1 / m1.sum()
            pbs = pbs_for_round2(c, cont, y0, y1)
            res = resolve(game, pbs, iterations=solve_iters)
            for hi, h in enumerate(LEDUC_CARDS):
                if hi == ci:
                    continue
                acc = 0.0
                for oi, o in enumerate(LEDUC_CARDS):
                    if oi in (hi, ci):
                        continue
                    deal = ((h, o, c) if traverser == 0 else (o, h, c)) \
                        + pbs.actions
                    ev = _ev_under_strategy(game, res.strategies, deal)
                    x_o = x_opp[oi]
                    acc += x_o * ev[traverser]
                u_direct[hi] += 0.25 * acc

        np.testing.assert_allclose(u_formula, u_direct, atol=1e-6)


class TestExactCFVs:

    def test_zero_sum_over_joint(self, game):
        """Joint-weighted totals of v0 and v1 must cancel (zero-sum game)."""
        rng = np.random.default_rng(3)
        comm = "Q2"
        ci = CARD_IDX[comm]
        y0 = rng.random(6); y0[ci] = 0.0; y0 /= y0.sum()
        y1 = rng.random(6); y1[ci] = 0.0; y1 /= y1.sum()
        v0, v1 = exact_round2_cfvs(game, comm, 3.0, y0, y1, solve_iters=100)

        # v_p is conditioned per holding; reconstruct joint expectations:
        # E[u0] = Σ_i P0(i) v0(i) with P0(i) ∝ y0(i)·Σ_{j∉{i,c}} y1(j)
        p0 = np.array([y0[i] * sum(y1[j] for j in range(6) if j not in (i, ci))
                       for i in range(6)])
        p1 = np.array([y1[i] * sum(y0[j] for j in range(6) if j not in (i, ci))
                       for i in range(6)])
        e0 = float((p0 / p0.sum()) @ v0)
        e1 = float((p1 / p1.sum()) @ v1)
        assert abs(e0 + e1) < 1e-6, f"E[u0]+E[u1] = {e0 + e1:.2e}"

    def test_fast_cfvs_match_deal_based_on_support(self, game):
        """Vector-CFR CFVs must match the deal-based solver where the
        holding has real mass. Off-support values are equilibrium-
        dependent (non-unique) and excluded — see cfv_net.train_cfv_net
        mass weighting."""
        from src.search.cfv import fast_round2_cfvs
        rng = np.random.default_rng(11)
        comm = "K2"
        ci = CARD_IDX[comm]
        y0 = rng.dirichlet(np.ones(6)); y0[ci] = 0; y0 /= y0.sum()
        y1 = rng.dirichlet(np.ones(6)); y1[ci] = 0; y1 /= y1.sum()
        v0s, v1s = exact_round2_cfvs(game, comm, 3.0, y0, y1,
                                     solve_iters=1500)
        v0f, v1f = fast_round2_cfvs(game, comm, 3.0, y0, y1,
                                    solve_iters=1500)
        m0 = y0 > 0.10
        m1 = y1 > 0.10
        np.testing.assert_allclose(v0f[m0], v0s[m0], atol=0.15)
        np.testing.assert_allclose(v1f[m1], v1s[m1], atol=0.15)

    def test_community_holding_zero(self, game):
        y = np.full(6, 1 / 5); y[CARD_IDX["K1"]] = 0.0
        v0, v1 = exact_round2_cfvs(game, "K1", 1.0, y, y, solve_iters=30)
        assert v0[CARD_IDX["K1"]] == 0.0
        assert v1[CARD_IDX["K1"]] == 0.0
