"""
Exact combo-level vector CFR for NLHE river subgames.

Stage B of the ReBeL extension (flop/turn depth-limited search): the
river is the exact base case — turn value-net targets are river subgame
root values, so the river solver's speed bounds the whole data pipeline.
The deal-based UnsafeSubgameSolver enumerates O(|r0|·|r1|) concrete
deals; this solver instead walks the PUBLIC river betting tree once per
iteration carrying 1326-dim per-combo reach vectors:

  * fold leaves:      u(h) = payoff · Σ_{h' disjoint} x_opp(h'),
                      card removal in O(1) per combo via per-card sums
                      (inclusion–exclusion: total − s[c1] − s[c2] + x(h)).
  * showdown leaves:  u(h) = inv · (weaker_mass(h) − stronger_mass(h)),
                      the DeepStack river trick — combos sorted by 7-card
                      strength once per solve, tie-group sweep with
                      per-card cumulative masses → exact card removal in
                      O(n) per leaf evaluation.

Infosets are per (public node, combo); regret matching with CFR+ clamp
and linear strategy averaging, alternating updates — the same recipe
validated on Leduc (src/search/vector_cfr.py).

The 50-bucket range compression (plan §Phase 3 task 1) applies to the
VALUE NET interface only (bucket_map / bucket_values below); the solver
itself is exact at combo level.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..abstraction.equity import evaluate_7card

N_COMBOS = 1326
COMBOS: list[tuple[int, int]] = [tuple(p) for p in combinations(range(52), 2)]
COMBO_IDX: dict[tuple[int, int], int] = {c: i for i, c in enumerate(COMBOS)}

# Combo → its two cards, as arrays for vectorised ops.
_C1 = np.asarray([c[0] for c in COMBOS], dtype=np.int64)
_C2 = np.asarray([c[1] for c in COMBOS], dtype=np.int64)


def combo_range_to_vector(range_dict: dict, board: tuple) -> np.ndarray:
    """{combo: prob} → normalised [1326] vector, board conflicts zeroed."""
    v = np.zeros(N_COMBOS)
    board_set = set(board)
    for combo, p in range_dict.items():
        key = tuple(sorted(combo))
        if p > 0 and not (board_set & set(key)):
            v[COMBO_IDX[key]] = p
    s = v.sum()
    if s <= 0:
        mask = np.asarray([not (board_set & set(c)) for c in COMBOS], float)
        return mask / mask.sum()
    return v / s


def _per_card_sums(x: np.ndarray) -> np.ndarray:
    """s[c] = Σ mass of combos containing card c. O(n)."""
    s = np.zeros(52)
    np.add.at(s, _C1, x)
    np.add.at(s, _C2, x)
    return s


def disjoint_mass(x_opp: np.ndarray) -> np.ndarray:
    """S_excl[h] = Σ_{h' disjoint from h} x_opp(h') for every combo h."""
    s = _per_card_sums(x_opp)
    return x_opp.sum() - s[_C1] - s[_C2] + x_opp


class RiverVectorCFR:
    """Vector CFR over the public river betting tree.

    Args:
        game:          PostflopNLHE (defines betting structure).
        node_history:  concrete history at a river decision point (only
                       its board + action prefix matter; hole cards in
                       the tuple are ignored).
        x0, x1:        [1326] range vectors for P0/P1 at the node
                       (board conflicts must already be zeroed).
        iterations:    CFR+ iterations.
    """

    def __init__(self, game, node_history, x0: np.ndarray, x1: np.ndarray,
                 iterations: int = 300):
        self.game = game
        self.board = tuple(node_history[2])
        self.base_actions = tuple(node_history[3:])
        self.roots = (np.asarray(x0, float).copy(),
                      np.asarray(x1, float).copy())
        self.iterations = iterations
        self._t = 0

        # Representative hole cards for public-tree queries (legality
        # depends only on actions, never on private cards).
        live = [c for c in range(52) if c not in self.board][:4]
        self._rep_holes = ((live[0], live[1]), (live[2], live[3]))

        # Board-conflict mask + 7-card strength order (once per solve).
        board_set = set(self.board)
        self.live_mask = np.asarray(
            [not (board_set & set(c)) for c in COMBOS], dtype=bool)
        scores = np.full(N_COMBOS, -1, dtype=np.int64)
        for i, c in enumerate(COMBOS):
            if self.live_mask[i]:
                scores[i] = evaluate_7card(tuple(c) + self.board)
        self.scores = scores
        self._order = np.argsort(scores, kind="stable")  # dead combos first
        self._n_dead = int((~self.live_mask).sum())

        self._regret: dict[tuple, np.ndarray] = {}
        self._strat_sum: dict[tuple, np.ndarray] = {}
        self._n_actions: dict[tuple, int] = {}

    # ── Public-tree helpers ──────────────────────────────────────────────────

    def _rep(self, cont_actions: tuple) -> tuple:
        return (self._rep_holes[0], self._rep_holes[1], self.board) \
            + self.base_actions + cont_actions

    def _tables(self, key: tuple, na: int):
        if key not in self._regret:
            self._regret[key] = np.zeros((N_COMBOS, na))
            self._strat_sum[key] = np.zeros((N_COMBOS, na))
            self._n_actions[key] = na
        return self._regret[key], self._strat_sum[key]

    @staticmethod
    def _strategy(regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        na = regret.shape[1]
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / na)

    # ── Terminal values ──────────────────────────────────────────────────────

    def _showdown_values(self, rep: tuple, x_opp: np.ndarray,
                         traverser: int) -> np.ndarray:
        """u[h] = inv · (weaker_mass(h) − stronger_mass(h)), exact removal."""
        state = self.game._parse_state(rep)
        inv = float(state["invested"][traverser])  # equal at showdown (HU)

        order = self._order[self._n_dead:]          # live combos, ascending
        sc = self.scores[order]
        x_sorted = x_opp[order]

        weaker = np.zeros(N_COMBOS)
        stronger = np.zeros(N_COMBOS)
        cum = 0.0
        cum_card = np.zeros(52)
        # forward sweep (weaker mass), grouped by tie value
        i = 0
        n = len(order)
        while i < n:
            j = i
            while j < n and sc[j] == sc[i]:
                j += 1
            idx = order[i:j]
            weaker[idx] = cum - cum_card[_C1[idx]] - cum_card[_C2[idx]]
            grp = x_sorted[i:j]
            cum += float(grp.sum())
            np.add.at(cum_card, _C1[idx], grp)
            np.add.at(cum_card, _C2[idx], grp)
            i = j
        # backward sweep (stronger mass)
        cum = 0.0
        cum_card[:] = 0.0
        i = n
        while i > 0:
            j = i
            while j > 0 and sc[j - 1] == sc[i - 1]:
                j -= 1
            idx = order[j:i]
            stronger[idx] = cum - cum_card[_C1[idx]] - cum_card[_C2[idx]]
            grp = x_sorted[j:i]
            cum += float(grp.sum())
            np.add.at(cum_card, _C1[idx], grp)
            np.add.at(cum_card, _C2[idx], grp)
            i = j
        u = inv * (weaker - stronger)
        u[~self.live_mask] = 0.0
        return u

    # ── Traversal ────────────────────────────────────────────────────────────

    def _traverse(self, cont: tuple, x_tr: np.ndarray, x_opp: np.ndarray,
                  traverser: int) -> np.ndarray:
        rep = self._rep(cont)
        game = self.game

        if game.is_terminal(rep):
            state = game._parse_state(rep)
            if state["folded"][0] or state["folded"][1]:
                f = float(game.terminal_payoffs(rep)[traverser])
                return f * disjoint_mass(x_opp)
            return self._showdown_values(rep, x_opp, traverser)

        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        na = len(legal)
        key = (cont,)
        regret, strat_sum = self._tables(key, na)
        sigma = self._strategy(regret)               # [1326, na]

        if player == traverser:
            u_a = np.zeros((na, N_COMBOS))
            for ai, a in enumerate(legal):
                u_a[ai] = self._traverse(cont + (a,), x_tr * sigma[:, ai],
                                         x_opp, traverser)
            u = np.einsum("ha,ah->h", sigma, u_a)
            regret += (u_a.T - u[:, None])
            np.maximum(regret, 0.0, out=regret)
            strat_sum += self._t * x_tr[:, None] * sigma
            return u
        u = np.zeros(N_COMBOS)
        for ai, a in enumerate(legal):
            u += self._traverse(cont + (a,), x_tr, x_opp * sigma[:, ai],
                                traverser)
        return u

    def solve(self) -> None:
        r0, r1 = self.roots
        for t in range(1, self.iterations + 1):
            self._t = t
            self._traverse((), r0.copy(), r1.copy(), 0)
            self._traverse((), r1.copy(), r0.copy(), 1)

    # ── Values & strategies ──────────────────────────────────────────────────

    def _avg_sigma(self, key: tuple, na: int) -> np.ndarray:
        ss = self._strat_sum.get(key)
        if ss is None:
            return np.full((N_COMBOS, na), 1.0 / na)
        tot = ss.sum(axis=1, keepdims=True)
        return np.where(tot > 0, ss / np.where(tot > 0, tot, 1.0), 1.0 / na)

    def _value_pass(self, cont: tuple, x_opp: np.ndarray,
                    traverser: int) -> np.ndarray:
        rep = self._rep(cont)
        game = self.game
        if game.is_terminal(rep):
            state = game._parse_state(rep)
            if state["folded"][0] or state["folded"][1]:
                f = float(game.terminal_payoffs(rep)[traverser])
                return f * disjoint_mass(x_opp)
            return self._showdown_values(rep, x_opp, traverser)
        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        sigma = self._avg_sigma((cont,), len(legal))
        if player == traverser:
            u = np.zeros(N_COMBOS)
            for ai, a in enumerate(legal):
                u += sigma[:, ai] * self._value_pass(cont + (a,), x_opp,
                                                     traverser)
            return u
        u = np.zeros(N_COMBOS)
        for ai, a in enumerate(legal):
            u += self._value_pass(cont + (a,), x_opp * sigma[:, ai],
                                  traverser)
        return u

    def root_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Conditional per-combo EVs (v0[1326], v1[1326]) under avg σ:
        v_p(h) = u_p(h) / Σ_{h' disjoint} x_opp(h'). Zero where the combo
        conflicts with the board or the opponent has no disjoint mass."""
        out = []
        for tr in (0, 1):
            x_opp = self.roots[1 - tr]
            u = self._value_pass((), x_opp.copy(), tr)
            denom = disjoint_mass(x_opp)
            v = np.zeros(N_COMBOS)
            ok = (denom > 1e-12) & self.live_mask
            v[ok] = u[ok] / denom[ok]
            out.append(v)
        return out[0], out[1]

    def strategy_at(self, history, player: int) -> np.ndarray:
        """Avg strategy for the player's ACTUAL combo at a continuation
        of the root node (history must extend node_history's prefix)."""
        cont = tuple(history[3:])[len(self.base_actions):]
        legal = self.game.legal_actions(history)
        sigma = self._avg_sigma((cont,), len(legal))
        combo = tuple(sorted(history[player]))
        return sigma[COMBO_IDX[combo]]


# ── Bucket interface for the value net ───────────────────────────────────────

def bucket_map(board: tuple, k: int = 50) -> np.ndarray:
    """combo → bucket index [1326], strength-percentile buckets among
    live combos (equal combo counts). Board-conflicting combos → -1."""
    board_set = set(board)
    live = [i for i, c in enumerate(COMBOS) if not (board_set & set(c))]
    scores = np.asarray([evaluate_7card(COMBOS[i] + tuple(board))
                         for i in live])
    order = np.argsort(scores, kind="stable")
    buckets = np.full(N_COMBOS, -1, dtype=np.int64)
    n = len(live)
    for rank, pos in enumerate(order):
        buckets[live[pos]] = min(k - 1, rank * k // n)
    return buckets


def bucket_range(x: np.ndarray, buckets: np.ndarray, k: int = 50) -> np.ndarray:
    """[1326] combo range → [k] bucket distribution (sums to 1)."""
    out = np.zeros(k)
    ok = buckets >= 0
    np.add.at(out, buckets[ok], x[ok])
    s = out.sum()
    return out / s if s > 0 else np.full(k, 1.0 / k)


def bucket_values(v: np.ndarray, x_own: np.ndarray, buckets: np.ndarray,
                  k: int = 50) -> np.ndarray:
    """Per-bucket EVs: own-range-weighted mean of per-combo values, with
    uniform fallback for empty buckets (mass floor keeps targets defined
    for every bucket — same off-support rationale as the Leduc net)."""
    num = np.zeros(k)
    den = np.zeros(k)
    ok = buckets >= 0
    w = x_own[ok] + 1e-9
    np.add.at(num, buckets[ok], w * v[ok])
    np.add.at(den, buckets[ok], w)
    out = np.zeros(k)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out
