"""
Round-2 PBS counterfactual values: exact targets + encoding.

A round-2 root PBS is fully described by (community card, per-player pot
contribution, both ranges): different round-1 action prefixes with the
same pot yield the identical round-2 game. This module provides

    exact_round2_cfvs()   per-holding expected values under an exact
                          CFR+ solve of the round-2 subgame — the
                          training targets for the PBS value net
                          (ReBeL Alg. 1 step: solve subgame, store values)

    encode_round2_pbs()   19-dim network input:
                          [comm card one-hot 6][pot contribution /5]
                          [range0 6][range1 6]
                          Comm is encoded as the CARD (6-way), not the
                          rank: with rank-only encoding, comm=Q1 vs Q2
                          with identical ranges are indistinguishable
                          inputs but have different value vectors (which
                          slot is dead / which pairs) — an irreducible
                          target conflict, measured as a holdout-MAE
                          plateau (~0.55 chips) despite training loss
                          → 0.10.

Value convention (matches VectorCFR's leaf evaluator contract):
    v_p(h) = E[u_p | p holds h, opponent ~ range_opp restricted to
             h' ∉ {h, comm} and renormalised, both play the solved σ*]
    v_p(comm_card) = 0 (impossible holding).
"""

from __future__ import annotations

import numpy as np

from ..games.leduc import LeducHoldem
from .pbs import LEDUC_CARDS, CARD_IDX, LeducPBS
from .resolve import resolve

ENCODING_DIMS = 19
# Synthetic round-1 prefixes producing each legal per-player contribution.
_PREFIX_FOR_CONT = {1.0: ("c", "c"), 3.0: ("r", "k"), 5.0: ("r", "r", "k")}


def pbs_for_round2(comm: str, cont: float,
                   y0: np.ndarray, y1: np.ndarray) -> LeducPBS:
    """Build a round-2 root PBS from its canonical parameters."""
    prefix = _PREFIX_FOR_CONT[float(cont)]
    return LeducPBS(
        actions=tuple(prefix),
        community=comm,
        ranges=(tuple(float(x) for x in y0), tuple(float(x) for x in y1)),
    )


def encode_round2_pbs(comm: str, cont: float,
                      y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    enc = np.zeros(ENCODING_DIMS, dtype=np.float32)
    enc[CARD_IDX[comm]] = 1.0
    enc[6] = float(cont) / 5.0
    enc[7:13] = np.asarray(y0, dtype=np.float32)
    enc[13:19] = np.asarray(y1, dtype=np.float32)
    return enc


def _ev_under_strategy(game: LeducHoldem, strategies: dict,
                       history: tuple) -> tuple[float, float]:
    """Expected (p0, p1) payoff of a concrete deal under avg strategies."""
    if game.is_terminal(history):
        p = game.terminal_payoffs(history)
        return float(p[0]), float(p[1])
    player = game.current_player(history)
    legal = game.legal_actions(history)
    key = game.info_set_key(history, player)
    probs = strategies.get(key)
    if probs is None:
        probs = np.ones(len(legal)) / len(legal)
    ev0 = ev1 = 0.0
    for prob, a in zip(probs, legal):
        if prob < 1e-12:
            continue
        e0, e1 = _ev_under_strategy(game, strategies, history + (a,))
        ev0 += prob * e0
        ev1 += prob * e1
    return ev0, ev1


def exact_round2_cfvs(
    game: LeducHoldem,
    comm: str,
    cont: float,
    y0: np.ndarray,
    y1: np.ndarray,
    solve_iters: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the round-2 subgame exactly; return (v0[6], v1[6]).

    Deal EVs are cached per (hero card, opp card) pair, then aggregated
    per holding with the opponent's range under card removal.
    """
    pbs = pbs_for_round2(comm, cont, y0, y1)
    res = resolve(game, pbs, iterations=solve_iters)
    strategies = res.strategies
    prefix = pbs.actions
    c_idx = CARD_IDX[comm]

    ev_pair: dict[tuple[int, int], tuple[float, float]] = {}

    def pair_ev(i: int, j: int) -> tuple[float, float]:
        if (i, j) not in ev_pair:
            h = (LEDUC_CARDS[i], LEDUC_CARDS[j], comm) + prefix
            ev_pair[(i, j)] = _ev_under_strategy(game, strategies, h)
        return ev_pair[(i, j)]

    v0 = np.zeros(6)
    v1 = np.zeros(6)
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)

    for i in range(6):
        if i == c_idx:
            continue
        # P0 holds i: opponent ~ y1 excluding {i, comm}
        w = y1.copy(); w[i] = 0.0; w[c_idx] = 0.0
        s = w.sum()
        if s > 0:
            v0[i] = sum(w[j] * pair_ev(i, j)[0] for j in range(6) if w[j] > 0) / s
        # P1 holds i: opponent ~ y0 excluding {i, comm}
        w = y0.copy(); w[i] = 0.0; w[c_idx] = 0.0
        s = w.sum()
        if s > 0:
            v1[i] = sum(w[j] * pair_ev(j, i)[1] for j in range(6) if w[j] > 0) / s

    return v0, v1


def fast_round2_cfvs(
    game: LeducHoldem,
    comm: str,
    cont: float,
    y0: np.ndarray,
    y1: np.ndarray,
    solve_iters: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector-CFR version of exact_round2_cfvs (~30× faster).

    Solves the round-2 subgame on the public tree with per-holding reach
    vectors, then converts root counterfactual values to conditional EVs:
    v_p(h) = u_p(h) / Σ_{h'∉{h,comm}} y_opp(h'). Validated to match the
    deal-based exact_round2_cfvs within solver tolerance.
    """
    from .vector_cfr import VectorCFR

    pbs = pbs_for_round2(comm, cont, y0, y1)
    solver = VectorCFR(game, pbs, leaf_evaluator=None,
                       iterations=solve_iters)
    solver.solve()
    c_idx = CARD_IDX[comm]
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)

    out = []
    for traverser, y_opp in ((0, y1), (1, y0)):
        u = solver.root_values(traverser)
        denom = y_opp.sum() - y_opp          # Σ excl h (comm already 0)
        v = np.zeros(6)
        mask = (denom > 1e-12)
        v[mask] = u[mask] / denom[mask]
        v[c_idx] = 0.0
        out.append(v)
    return out[0], out[1]


def exact_leaf_evaluator(game: LeducHoldem, solve_iters: int = 200):
    """Oracle leaf evaluator for VectorCFR (validation / target generation)."""
    def evaluator(comm, cont, y0, y1):
        return exact_round2_cfvs(game, comm, cont, y0, y1,
                                 solve_iters=solve_iters)
    return evaluator
