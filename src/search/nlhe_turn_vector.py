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
    bucket_map, bucket_range, disjoint_mass,
)
from .nlhe_cfv import K_BUCKETS, encode_river_pbs
from ..abstraction.equity import evaluate_7card

TURN_STREET = 2


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

        # Per-river-card precomputation (once per solve):
        #   combo mask, bucket map, and (for all-in leaves) strength order.
        self._card_mask: dict[int, np.ndarray] = {}
        self._bmaps: dict[int, np.ndarray] = {}
        self._scores: dict[int, np.ndarray] = {}
        for c in self.river_cards:
            mask = self.live_mask & (_C1 != c) & (_C2 != c)
            self._card_mask[c] = mask
            board5 = self.turn_board + (c,)
            self._bmaps[c] = bucket_map(board5, K_BUCKETS)
            sc = np.full(N_COMBOS, -1, dtype=np.int64)
            idx = np.where(mask)[0]
            for i in idx:
                sc[i] = evaluate_7card(tuple(COMBOS[i]) + board5)
            self._scores[c] = sc

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
        """Exact equity over the 46 runouts, sorted-sweep per card."""
        inv = float(self.game._parse_state(rep)["invested"][traverser])
        u = np.zeros(N_COMBOS)
        p = 1.0 / len(self.river_cards)
        for c in self.river_cards:
            mask = self._card_mask[c]
            sc = self._scores[c]
            xm = np.where(mask, x_opp, 0.0)
            order = np.argsort(sc, kind="stable")
            order = order[sc[order] >= 0]
            u_c = self._sweep(order, sc, xm, inv)
            u_c[~mask] = 0.0
            u += p * u_c
        return u

    @staticmethod
    def _sweep(order, sc, x_sorted_src, inv) -> np.ndarray:
        weaker = np.zeros(N_COMBOS)
        stronger = np.zeros(N_COMBOS)
        for direction in (1, -1):
            cum = 0.0
            cum_card = np.zeros(52)
            seq = order if direction == 1 else order[::-1]
            svals = sc[seq]
            i, n = 0, len(seq)
            out = weaker if direction == 1 else stronger
            while i < n:
                j = i
                while j < n and svals[j] == svals[i]:
                    j += 1
                idx = seq[i:j]
                out[idx] = cum - cum_card[_C1[idx]] - cum_card[_C2[idx]]
                grp = x_sorted_src[idx]
                cum += float(grp.sum())
                np.add.at(cum_card, _C1[idx], grp)
                np.add.at(cum_card, _C2[idx], grp)
                i = j
        return inv * (weaker - stronger)

    def _river_boundary(self, rep: tuple, x_tr: np.ndarray,
                        x_opp: np.ndarray, traverser: int) -> np.ndarray:
        """Depth limit: batched net query over all 46 river cards."""
        pot = float(self.game._parse_state(rep)["pot"])
        stack = float(self.game.starting_stack)

        encs, metas = [], []
        for c in self.river_cards:
            mask = self._card_mask[c]
            m0 = np.where(mask, x_tr if traverser == 0 else x_opp, 0.0)
            m1 = np.where(mask, x_opp if traverser == 0 else x_tr, 0.0)
            s0, s1 = m0.sum(), m1.sum()
            if s0 <= 0 or s1 <= 0:
                continue
            bm = self._bmaps[c]
            encs.append(encode_river_pbs(
                self.turn_board + (c,), pot, stack,
                bucket_range(m0 / s0, bm, K_BUCKETS),
                bucket_range(m1 / s1, bm, K_BUCKETS)))
            metas.append((c, bm))
        if not encs:
            return np.zeros(N_COMBOS)

        with torch.no_grad():
            out = self.net(torch.as_tensor(np.stack(encs), dtype=torch.float32,
                                           device=self.device)).cpu().numpy()
        out = out * pot                     # un-normalise (targets are v/pot)

        u = np.zeros(N_COMBOS)
        p = 1.0 / len(self.river_cards)
        half = K_BUCKETS if traverser == 1 else 0
        for (c, bm), row in zip(metas, out):
            v_bucket = row[half:half + K_BUCKETS]
            mask = self._card_mask[c]
            v_combo = np.zeros(N_COMBOS)
            ok = mask & (bm >= 0)
            v_combo[ok] = v_bucket[bm[ok]]
            x_o = np.where(mask, x_opp, 0.0)
            u += p * v_combo * disjoint_mass(x_o)
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
