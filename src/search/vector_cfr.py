"""
Vector-form CFR over the Leduc public tree (DeepStack/ReBeL style).

Unlike CFRSolver (which traverses concrete deals), this solver walks the
PUBLIC tree once per iteration carrying per-holding reach vectors. That is
the form depth-limited solving requires: at a depth-limit leaf the value
depends on the CURRENT iteration's ranges (a PBS), which a fixed
terminal_payoffs interface cannot express.

Two leaf modes:

    expand (leaf_evaluator=None)
        The community-reveal boundary is expanded as an explicit chance
        node (6 branches, card removal via reach masking) and round 2 is
        solved to terminal. This mode is exact and exists to validate the
        vector-CFR machinery against the Phase 1 deal-based resolve().

    evaluate (leaf_evaluator given)
        The tree is truncated at the reveal boundary. For each community
        card the evaluator is queried with the leaf PBS (community, pot
        contribution, both normalised ranges) and must return per-holding
        expected values (v0[6], v1[6]) under the round-2 equilibrium —
        conditioned on the holding, opponent distributed by the given
        range with card removal. ReBeL Algorithm 1 leaf substitution.

Conventions:
    - Holdings are the 6 concrete cards; infosets are RANK-level (J1/J2
      share regret/strategy tables, matching LeducHoldem.info_set_key).
    - Traversal values are unnormalised counterfactual values:
      u_p(h) = sum over opponent holdings h' (and community) of
      opp_reach(h') * P(chance) * payoff_p. Alternating updates, CFR+
      regret clamp, linear strategy averaging.
    - Strategy keys match LeducHoldem.info_set_key ("RANK|COMM|actions")
      so results plug straight into the existing carriers and agents.
"""

from __future__ import annotations

import numpy as np

from ..games.leduc import LeducHoldem, RANKS
from .pbs import LEDUC_CARDS, CARD_IDX, LeducPBS, representative_history

N_CARDS = 6
_RANK_OF = np.array([0, 0, 1, 1, 2, 2])          # card idx → rank idx
_RANK_CARDS = [(0, 1), (2, 3), (4, 5)]           # rank idx → card idxs


def _fold_payoff(game: LeducHoldem, actions: tuple[str, ...],
                 player: int) -> float:
    """Payoff for `player` at a fold-terminal action sequence (holdings
    are irrelevant at folds)."""
    payoffs = game.terminal_payoffs(("J1", "Q1", "K1") + actions)
    return float(payoffs[player])


def _showdown_matrix(game: LeducHoldem, actions: tuple[str, ...],
                     comm: str, player: int) -> np.ndarray:
    """U[h, h'] = payoff for `player` holding card h vs opponent card h'
    at a showdown-terminal sequence with community `comm`. Diagonal and
    community rows/cols are zero (impossible deals)."""
    U = np.zeros((N_CARDS, N_CARDS))
    for i, ci in enumerate(LEDUC_CARDS):
        if ci == comm:
            continue
        for j, cj in enumerate(LEDUC_CARDS):
            if j == i or cj == comm:
                continue
            h = (ci, cj, comm) if player == 0 else (cj, ci, comm)
            U[i, j] = game.terminal_payoffs(h + actions)[player]
    return U


class VectorCFR:
    """Vector CFR rooted at a pre-reveal Leduc PBS."""

    def __init__(
        self,
        game: LeducHoldem,
        pbs: LeducPBS,
        leaf_evaluator=None,
        iterations: int = 400,
    ):
        if pbs.community is not None and leaf_evaluator is not None:
            raise ValueError(
                "depth-limited mode roots at pre-reveal public states; "
                "post-reveal roots are exact by construction"
            )
        self.game = game
        self.root_actions = pbs.actions
        self.root_comm = pbs.community
        self.root_ranges = (pbs.range_array(0), pbs.range_array(1))
        self.leaf_evaluator = leaf_evaluator
        self.iterations = iterations
        # (info_key, ) → per-rank tables, shape [3, n_actions]
        self._regret: dict[str, np.ndarray] = {}
        self._strat_sum: dict[str, np.ndarray] = {}
        self._n_actions: dict[str, int] = {}
        self._t = 0
        # Terminal caches: payoff matrices are pure functions of the
        # public sequence (+comm), recomputed thousands of times per
        # solve without memoisation.
        self._fold_cache: dict[tuple, float] = {}
        self._showdown_cache: dict[tuple, np.ndarray] = {}

    # ── Infoset helpers ──────────────────────────────────────────────────────

    def _node_key(self, actions: tuple[str, ...], comm: str | None) -> str:
        """Public part of the infoset key: 'COMM|actions' (rank prepended
        per-holding on output)."""
        _, _, r1_done = self.game._split_rounds(actions)
        comm_rank = comm[0] if (r1_done and comm is not None) else ""
        return f"{comm_rank}|{''.join(actions)}"

    def _tables(self, node_key: str, n_actions: int):
        if node_key not in self._regret:
            self._regret[node_key] = np.zeros((3, n_actions))
            self._strat_sum[node_key] = np.zeros((3, n_actions))
            self._n_actions[node_key] = n_actions
        return self._regret[node_key], self._strat_sum[node_key]

    @staticmethod
    def _rank_strategy(regret: np.ndarray) -> np.ndarray:
        """Regret matching per rank: [3, na] → [3, na] probabilities."""
        pos = np.maximum(regret, 0.0)
        total = pos.sum(axis=1, keepdims=True)
        na = regret.shape[1]
        out = np.where(total > 0, pos / np.where(total > 0, total, 1.0),
                       1.0 / na)
        return out

    @staticmethod
    def _to_holdings(rank_mat: np.ndarray) -> np.ndarray:
        """[3, na] rank-level → [6, na] holding-level."""
        return rank_mat[_RANK_OF]

    # ── Traversal ────────────────────────────────────────────────────────────

    def _traverse(
        self,
        actions: tuple[str, ...],
        comm: str | None,
        x_tr: np.ndarray,     # traverser reach [6]
        x_opp: np.ndarray,    # opponent reach [6]
        traverser: int,
    ) -> np.ndarray:
        """Returns unnormalised counterfactual values u[6] for traverser."""
        game = self.game

        if game.is_terminal(("J1", "Q1", "K1") + actions):
            if "f" in actions:
                fk = (actions, traverser)
                if fk not in self._fold_cache:
                    self._fold_cache[fk] = _fold_payoff(game, actions,
                                                        traverser)
                f = self._fold_cache[fk]
                s = x_opp.sum()
                return f * (s - x_opp)
            # Showdown (round 2 complete, comm must be set in expand mode)
            sk = (actions, comm, traverser)
            if sk not in self._showdown_cache:
                self._showdown_cache[sk] = _showdown_matrix(
                    game, actions, comm, traverser)
            return self._showdown_cache[sk] @ x_opp

        _, _, r1_done = game._split_rounds(actions)
        if r1_done and comm is None:
            return self._reveal_boundary(actions, x_tr, x_opp, traverser)

        rep = representative_history("J1", 0, comm, actions)
        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        na = len(legal)
        node_key = self._node_key(actions, comm)
        regret, strat_sum = self._tables(node_key, na)
        sigma_rank = self._rank_strategy(regret)
        sigma = self._to_holdings(sigma_rank)          # [6, na]

        if player == traverser:
            u_a = np.zeros((na, N_CARDS))
            for ai, a in enumerate(legal):
                u_a[ai] = self._traverse(
                    actions + (a,), comm, x_tr * sigma[:, ai], x_opp,
                    traverser,
                )
            u = (sigma.T * u_a).sum(axis=0)            # [6]
            # Regret update (rank-level, CFR+ clamp)
            diff = u_a - u[None, :]                    # [na, 6]
            for r, cards in enumerate(_RANK_CARDS):
                regret[r] += diff[:, cards].sum(axis=1)
            np.maximum(regret, 0.0, out=regret)
            # Linear strategy averaging weighted by own reach
            w = self._t
            for r, cards in enumerate(_RANK_CARDS):
                strat_sum[r] += w * x_tr[list(cards)].sum() * sigma_rank[r]
            return u
        else:
            u = np.zeros(N_CARDS)
            for ai, a in enumerate(legal):
                u += self._traverse(
                    actions + (a,), comm, x_tr, x_opp * sigma[:, ai],
                    traverser,
                )
            return u

    def _reveal_boundary(
        self,
        actions: tuple[str, ...],
        x_tr: np.ndarray,
        x_opp: np.ndarray,
        traverser: int,
    ) -> np.ndarray:
        """Chance node: community reveal. Expand or evaluate."""
        u = np.zeros(N_CARDS)

        if self.leaf_evaluator is None:
            # Exact expansion: each community card with P = 1/4 given a
            # consistent deal; masking handles card removal.
            for c_idx, c in enumerate(LEDUC_CARDS):
                m_tr = x_tr.copy();  m_tr[c_idx] = 0.0
                m_opp = x_opp.copy(); m_opp[c_idx] = 0.0
                u_c = self._traverse(actions, c, m_tr, m_opp, traverser)
                u_c[c_idx] = 0.0     # holding == community is impossible
                u += 0.25 * u_c
            return u

        # Depth limit: query the leaf evaluator per community card.
        raises = actions.count("r")
        cont = 1.0 + 2.0 * raises          # per-player pot contribution
        for c_idx, c in enumerate(LEDUC_CARDS):
            x0 = (x_tr if traverser == 0 else x_opp).copy()
            x1 = (x_opp if traverser == 0 else x_tr).copy()
            x0[c_idx] = 0.0
            x1[c_idx] = 0.0
            s0, s1 = x0.sum(), x1.sum()
            if s0 <= 0.0 or s1 <= 0.0:
                continue
            y0, y1 = x0 / s0, x1 / s1
            v0, v1 = self.leaf_evaluator(c, cont, y0, y1)
            v_tr = np.asarray(v0 if traverser == 0 else v1, dtype=np.float64)
            x_o = x1 if traverser == 0 else x0
            s_o = x_o.sum()
            contrib = 0.25 * v_tr * (s_o - x_o)
            contrib[c_idx] = 0.0
            u += contrib
        return u

    # ── Public API ───────────────────────────────────────────────────────────

    def solve(self) -> None:
        r0, r1 = self.root_ranges
        for t in range(1, self.iterations + 1):
            self._t = t
            self._traverse(self.root_actions, self.root_comm,
                           r0.copy(), r1.copy(), 0)
            self._traverse(self.root_actions, self.root_comm,
                           r1.copy(), r0.copy(), 1)

    def average_strategies(self) -> dict[str, np.ndarray]:
        """{LeducHoldem.info_set_key: probs} for every visited infoset."""
        out: dict[str, np.ndarray] = {}
        for node_key, ss in self._strat_sum.items():
            comm_rank, action_str = node_key.split("|", 1)
            na = self._n_actions[node_key]
            for r, rank in enumerate(RANKS):
                total = ss[r].sum()
                probs = ss[r] / total if total > 0 else np.full(na, 1.0 / na)
                out[f"{rank}|{comm_rank}|{action_str}"] = probs
        return out

    def root_values(self, traverser: int) -> np.ndarray:
        """Unnormalised per-holding counterfactual values at the root
        under the AVERAGE strategies (call after solve()). Exact-expand
        mode only. Divide by (Σ x_opp − x_opp[h]) for conditional EVs."""
        if self.leaf_evaluator is not None:
            raise ValueError("root_values requires exact expand mode")
        strategies = self.average_strategies()
        r0, r1 = self.root_ranges
        x_tr = (r0 if traverser == 0 else r1).copy()
        x_opp = (r1 if traverser == 0 else r0).copy()
        return self._value_pass(self.root_actions, self.root_comm,
                                x_opp, traverser, strategies)

    def _value_pass(self, actions, comm, x_opp, traverser,
                    strategies) -> np.ndarray:
        """u_tr[6] under fixed average strategies (no updates)."""
        game = self.game
        if game.is_terminal(("J1", "Q1", "K1") + actions):
            if "f" in actions:
                fk = (actions, traverser)
                if fk not in self._fold_cache:
                    self._fold_cache[fk] = _fold_payoff(game, actions,
                                                        traverser)
                s = x_opp.sum()
                return self._fold_cache[fk] * (s - x_opp)
            sk = (actions, comm, traverser)
            if sk not in self._showdown_cache:
                self._showdown_cache[sk] = _showdown_matrix(
                    game, actions, comm, traverser)
            return self._showdown_cache[sk] @ x_opp

        _, _, r1_done = game._split_rounds(actions)
        if r1_done and comm is None:
            u = np.zeros(N_CARDS)
            for c_idx, c in enumerate(LEDUC_CARDS):
                m_opp = x_opp.copy(); m_opp[c_idx] = 0.0
                u_c = self._value_pass(actions, c, m_opp, traverser,
                                       strategies)
                u_c[c_idx] = 0.0
                u += 0.25 * u_c
            return u

        rep = representative_history("J1", 0, comm, actions)
        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        node_key = self._node_key(actions, comm)
        comm_rank, action_str = node_key.split("|", 1)
        sigma = np.stack([
            strategies.get(f"{rank}|{comm_rank}|{action_str}",
                           np.full(len(legal), 1.0 / len(legal)))
            for rank in RANKS
        ])[_RANK_OF]                                     # [6, na]

        if player == traverser:
            u = np.zeros(N_CARDS)
            for ai, a in enumerate(legal):
                u += sigma[:, ai] * self._value_pass(
                    actions + (a,), comm, x_opp, traverser, strategies)
            return u
        u = np.zeros(N_CARDS)
        for ai, a in enumerate(legal):
            u += self._value_pass(actions + (a,), comm,
                                  x_opp * sigma[:, ai], traverser,
                                  strategies)
        return u

    def strategy_at(self, history, player: int) -> np.ndarray:
        key = self.game.info_set_key(history, player)
        strategies = self.average_strategies()
        n = len(self.game.legal_actions(history))
        if key in strategies:
            return strategies[key][:n]
        return np.ones(n) / n
