"""
Depth-limited turn vector CFR: turn betting exact, river leaves from the
CFV net (Stage B3 — the NLHE analogue of Leduc's VectorCFR evaluate mode).

Tree rooted at a turn decision node, per-combo (1326-dim) reach vectors:

  * fold leaves            exact (disjoint_mass, as in the river solver)
  * all-in showdowns       exact: expand the 46 river cards, per-card
                           strength sweep (no betting left, pure equity)
  * turn-street completion river chance node: for each of the 46 river
                           cards, mask reaches, bucket both ranges on the
                           now-5-card board, query the river CFV net
                           (all 46 cards in ONE batched forward), map
                           per-bucket values back to combos and convert
                           to unnormalised counterfactual form:
                               u(h) += 1/46 · v̂_c(h) · D_c(h)
                           where D_c(h) is the opponent's disjoint mass
                           under river card c — the identity validated
                           brute-force on Leduc (test_vector_cfr.py::
                           TestLeafConversion).

The evaluator contract matches nlhe_cfv.river_target training targets:
pot-normalised per-bucket conditional EVs → multiplied back by pot here.
"""

from __future__ import annotations

import numpy as np
import torch

from .nlhe_river_vector import (
    COMBOS, N_COMBOS, _C1, _C2,
    disjoint_mass,
)
from .nlhe_cfv import K_BUCKETS
from ..abstraction.equity import evaluate_7card

TURN_STREET = 2

# Combo → card incidence [1326, 52] for BLAS per-card sums.
_INC = np.zeros((N_COMBOS, 52))
_INC[np.arange(N_COMBOS), _C1] = 1.0
_INC[np.arange(N_COMBOS), _C2] = 1.0

# Pairwise card-overlap mask [1326, 1326]: True where combos share a card.
_OVERLAP = ((_C1[:, None] == _C1[None, :]) | (_C1[:, None] == _C2[None, :])
            | (_C2[:, None] == _C1[None, :]) | (_C2[:, None] == _C2[None, :]))

_BOARD_TABLE_CACHE: dict[tuple, tuple] = {}


def _board_tables(turn_board: tuple, river_cards: tuple, live_mask):
    """Per-turn-board tables (LRU-ish: unbounded, entries ~3 MB, and a
    training/eval run touches only a bounded set of boards)."""
    key = turn_board
    if key in _BOARD_TABLE_CACHE:
        return _BOARD_TABLE_CACHE[key]

    n_r = len(river_cards)
    card_mask: dict[int, np.ndarray] = {}
    bmaps: dict[int, np.ndarray] = {}
    scores: dict[int, np.ndarray] = {}
    mask_matrix = np.zeros((n_r, N_COMBOS), dtype=bool)
    bm_matrix = np.full((n_r, N_COMBOS), -1, dtype=np.int64)

    for r, c in enumerate(river_cards):
        mask = live_mask & (_C1 != c) & (_C2 != c)
        card_mask[c] = mask
        board5 = turn_board + (c,)
        sc = np.full(N_COMBOS, -1, dtype=np.int64)
        idx = np.where(mask)[0]
        for i in idx:
            sc[i] = evaluate_7card(tuple(COMBOS[i]) + board5)
        scores[c] = sc
        # Strength-percentile buckets from the SAME score array
        # (bucket_map would recompute every evaluate_7card).
        order = idx[np.argsort(sc[idx], kind="stable")]
        bm = np.full(N_COMBOS, -1, dtype=np.int64)
        n_live = len(order)
        bm[order] = np.minimum(K_BUCKETS - 1,
                               np.arange(n_live) * K_BUCKETS // n_live)
        bmaps[c] = bm
        mask_matrix[r] = mask
        bm_matrix[r] = bm

    from .nlhe_cfv import board_texture
    tex_matrix = np.asarray(
        [board_texture(turn_board + (c,)) for c in river_cards])

    # All-in showdown as ONE linear operator (the leaf value is linear
    # in the opponent reach): A[h,h'] = 1/R · Σ_{c: both alive under c}
    # sign(score_c(h) − score_c(h')), overlapping pairs zeroed. Profiling
    # showed the per-iteration tie-group sweeps were 95% of solve time;
    # this replaces them with a single BLAS matvec per leaf call.
    A = np.zeros((N_COMBOS, N_COMBOS), dtype=np.float32)
    for c in river_cards:
        sc = scores[c]
        alive = card_mask[c]
        sign = np.sign(sc[:, None] - sc[None, :]).astype(np.float32)
        pair_alive = np.outer(alive, alive)
        A += sign * pair_alive
    A[_OVERLAP] = 0.0
    A /= float(n_r)

    out = (card_mask, bmaps, scores, mask_matrix, bm_matrix, tex_matrix, A)
    _BOARD_TABLE_CACHE[key] = out
    return out


class TurnVectorCFR:
    """Depth-limited CFR at a turn decision node.

    Args:
        game:          PostflopNLHE.
        node_history:  concrete history at a turn decision (street 2).
                       history[2] holds the full 5-card board; only the
                       first 4 cards are public at the turn — the solver
                       marginalises over all 46 possible river cards.
        x0, x1:        [1326] combo ranges (turn-board conflicts zeroed).
        cfv_net:       river CFV net (CFVNet, 104 → 100).
        iterations:    CFR+ iterations.
    """

    def __init__(self, game, node_history, x0: np.ndarray, x1: np.ndarray,
                 cfv_net, iterations: int = 200, device: str = "cpu"):
        self.game = game
        self.turn_board = tuple(node_history[2][:4])
        self.base_actions = tuple(node_history[3:])
        self.roots = (np.asarray(x0, float).copy(),
                      np.asarray(x1, float).copy())
        self.net = cfv_net
        self.device = device
        self.iterations = iterations
        self._t = 0

        used = set(self.turn_board)
        self.river_cards = [c for c in range(52) if c not in used]
        live = [c for c in range(52) if c not in used][:4]
        self._rep_holes = ((live[0], live[1]), (live[2], live[3]))
        board_set = set(self.turn_board)
        self.live_mask = np.asarray(
            [not (board_set & set(c)) for c in COMBOS], dtype=bool)

        # Per-river-card precomputation, cached per turn board: 7-card
        # scores per (river card, combo) are pure functions of the board
        # and were the dominant setup cost (46 × 1326 evaluate_7card ≈
        # 6 s). Buckets are derived from the same score arrays instead
        # of recomputing them inside bucket_map (was double work).
        (self._card_mask, self._bmaps, self._scores,
         self._mask_matrix, self._bm_matrix, self._tex_matrix,
         self._allin_op) = \
            _board_tables(self.turn_board, tuple(self.river_cards),
                          self.live_mask)

        self._regret: dict[tuple, np.ndarray] = {}
        self._strat_sum: dict[tuple, np.ndarray] = {}

    # ── Public-tree helpers ──────────────────────────────────────────────────

    def _rep(self, cont: tuple) -> tuple:
        # Full 5-card board needed for a syntactically valid history; the
        # 5th card is arbitrary (turn-street legality ignores it).
        board5 = self.turn_board + (self.river_cards[0],)
        return (self._rep_holes[0], self._rep_holes[1], board5) \
            + self.base_actions + cont

    def _tables(self, key: tuple, na: int):
        if key not in self._regret:
            self._regret[key] = np.zeros((N_COMBOS, na))
            self._strat_sum[key] = np.zeros((N_COMBOS, na))
        return self._regret[key], self._strat_sum[key]

    @staticmethod
    def _strategy(regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        na = regret.shape[1]
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / na)

    # ── Leaves ───────────────────────────────────────────────────────────────

    def _allin_showdown(self, rep: tuple, x_opp: np.ndarray,
                        traverser: int) -> np.ndarray:
        """Exact equity over the 46 runouts: one precomputed matvec."""
        inv = float(self.game._parse_state(rep)["invested"][traverser])
        return inv * (self._allin_op @ x_opp.astype(np.float32)
                      ).astype(np.float64)

    def _river_boundary(self, rep: tuple, x_tr: np.ndarray,
                        x_opp: np.ndarray, traverser: int) -> np.ndarray:
        """Depth limit: one batched net query over all 46 river cards.

        Fully vectorised across cards: masked reaches [R,1326], bucket
        ranges via one flat scatter, per-card disjoint masses via a BLAS
        product with the combo→card incidence matrix. Semantics are
        identical to the original per-card loop (equality-tested).
        """
        pot = float(self.game._parse_state(rep)["pot"])
        stack = float(self.game.starting_stack)
        M = self._mask_matrix                       # [R, 1326] bool
        R = M.shape[0]

        x0 = x_tr if traverser == 0 else x_opp
        x1 = x_opp if traverser == 0 else x_tr
        X0 = M * x0[None, :]
        X1 = M * x1[None, :]
        s0 = X0.sum(axis=1)
        s1 = X1.sum(axis=1)
        ok_rows = (s0 > 0) & (s1 > 0)
        if not ok_rows.any():
            return np.zeros(N_COMBOS)

        # Bucket ranges for all cards in one scatter per player.
        bmf = self._bm_matrix                       # [R, 1326], -1 = dead
        valid = bmf >= 0
        flat = (bmf + np.arange(R)[:, None] * K_BUCKETS)[valid]
        br0 = np.zeros(R * K_BUCKETS)
        br1 = np.zeros(R * K_BUCKETS)
        np.add.at(br0, flat, (X0 / np.where(s0 > 0, s0, 1.0)[:, None])[valid])
        np.add.at(br1, flat, (X1 / np.where(s1 > 0, s1, 1.0)[:, None])[valid])
        br0 = br0.reshape(R, K_BUCKETS)
        br1 = br1.reshape(R, K_BUCKETS)

        encs = np.zeros((R, 4 + 2 * K_BUCKETS), dtype=np.float32)
        encs[:, 0] = pot / (2.0 * stack)
        encs[:, 1:4] = self._tex_matrix
        encs[:, 4:4 + K_BUCKETS] = br0
        encs[:, 4 + K_BUCKETS:] = br1

        with torch.no_grad():
            out = self.net(torch.as_tensor(encs[ok_rows],
                                           dtype=torch.float32,
                                           device=self.device)).cpu().numpy()
        out = out * pot                     # un-normalise (targets are v/pot)

        half = K_BUCKETS if traverser == 1 else 0
        v_bucket = np.zeros((R, K_BUCKETS))
        v_bucket[ok_rows] = out[:, half:half + K_BUCKETS]

        # Per-combo values back from buckets; dead entries stay 0.
        v_combo = np.take_along_axis(
            v_bucket, np.where(valid, bmf, 0), axis=1)
        v_combo[~valid] = 0.0

        # Disjoint opponent mass per (card, combo): BLAS per-card sums.
        Xop = M * x_opp[None, :]
        s_cards = Xop @ _INC                        # [R, 52]
        D = (Xop.sum(axis=1)[:, None]
             - s_cards[:, _C1] - s_cards[:, _C2] + Xop)

        u = (v_combo * D * ok_rows[:, None]).sum(axis=0) / R
        return u

    # ── Traversal ────────────────────────────────────────────────────────────

    def _traverse(self, cont: tuple, x_tr: np.ndarray, x_opp: np.ndarray,
                  traverser: int, averaging: bool = True) -> np.ndarray:
        rep = self._rep(cont)
        game = self.game
        state = game._parse_state(rep)

        if game.is_terminal(rep) or state["street_idx"] > TURN_STREET:
            if state["folded"][0] or state["folded"][1]:
                f = float(game.terminal_payoffs(rep)[traverser])
                return f * disjoint_mass(x_opp)
            if state["all_in"]:
                return self._allin_showdown(rep, x_opp, traverser)
            # Turn betting completed → river boundary (depth limit).
            return self._river_boundary(rep, x_tr, x_opp, traverser)

        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        na = len(legal)
        key = (cont,)
        regret, strat_sum = self._tables(key, na)
        sigma = self._strategy(regret)

        if player == traverser:
            u_a = np.zeros((na, N_COMBOS))
            for ai, a in enumerate(legal):
                u_a[ai] = self._traverse(cont + (a,), x_tr * sigma[:, ai],
                                         x_opp, traverser, averaging)
            u = np.einsum("ha,ah->h", sigma, u_a)
            if averaging:
                regret += (u_a.T - u[:, None])
                np.maximum(regret, 0.0, out=regret)
                strat_sum += self._t * x_tr[:, None] * sigma
            return u
        u = np.zeros(N_COMBOS)
        for ai, a in enumerate(legal):
            u += self._traverse(cont + (a,), x_tr, x_opp * sigma[:, ai],
                                traverser, averaging)
        return u

    def solve(self) -> None:
        r0, r1 = self.roots
        for t in range(1, self.iterations + 1):
            self._t = t
            self._traverse((), r0.copy(), r1.copy(), 0)
            self._traverse((), r1.copy(), r0.copy(), 1)

    def _avg_sigma(self, cont: tuple, na: int) -> np.ndarray:
        ss = self._strat_sum.get((cont,))
        if ss is None:
            return np.full((N_COMBOS, na), 1.0 / na)
        tot = ss.sum(axis=1, keepdims=True)
        return np.where(tot > 0, ss / np.where(tot > 0, tot, 1.0), 1.0 / na)

    def _value_pass(self, cont: tuple, x_tr: np.ndarray, x_opp: np.ndarray,
                    traverser: int) -> np.ndarray:
        """u_tr[1326] at the turn root under the AVERAGE strategies — same
        leaves as _traverse (fold / all-in / river-boundary net), no
        regret/strategy updates. Feeds turn-root CFVs for a turn CFV net."""
        rep = self._rep(cont)
        game = self.game
        state = game._parse_state(rep)
        if game.is_terminal(rep) or state["street_idx"] > TURN_STREET:
            if state["folded"][0] or state["folded"][1]:
                f = float(game.terminal_payoffs(rep)[traverser])
                return f * disjoint_mass(x_opp)
            if state["all_in"]:
                return self._allin_showdown(rep, x_opp, traverser)
            return self._river_boundary(rep, x_tr, x_opp, traverser)

        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        sigma = self._avg_sigma(cont, len(legal))
        if player == traverser:
            u = np.zeros(N_COMBOS)
            for ai, a in enumerate(legal):
                u += sigma[:, ai] * self._value_pass(
                    cont + (a,), x_tr * sigma[:, ai], x_opp, traverser)
            return u
        u = np.zeros(N_COMBOS)
        for ai, a in enumerate(legal):
            u += self._value_pass(cont + (a,), x_tr, x_opp * sigma[:, ai],
                                  traverser)
        return u

    def root_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Conditional per-combo turn-root EVs (v0[1326], v1[1326]) under
        the average strategies: v_p(h) = u_p(h) / Σ_{h' disjoint} x_opp(h').
        Zero where the combo conflicts with the turn board or the opponent
        has no disjoint mass. Mirrors RiverVectorCFR.root_values."""
        out = []
        for tr in (0, 1):
            x_tr = self.roots[tr]
            x_opp = self.roots[1 - tr]
            u = self._value_pass((), x_tr.copy(), x_opp.copy(), tr)
            denom = disjoint_mass(x_opp)
            v = np.zeros(N_COMBOS)
            ok = (denom > 1e-12) & self.live_mask
            v[ok] = u[ok] / denom[ok]
            out.append(v)
        return out[0], out[1]

    def strategy_at(self, history, player: int) -> np.ndarray:
        from .nlhe_river_vector import COMBO_IDX
        cont = tuple(history[3:])[len(self.base_actions):]
        legal = self.game.legal_actions(history)
        ss = self._strat_sum.get((cont,))
        if ss is None:
            return np.ones(len(legal)) / len(legal)
        combo = tuple(sorted(history[player]))
        row = ss[COMBO_IDX[combo], :len(legal)]
        s = row.sum()
        return row / s if s > 1e-12 else np.ones(len(legal)) / len(legal)
