"""
Depth-limited continual re-solving agent for Leduc (ReBeL Alg. 1).

Same public-tree walk as resolve.build_resolving_strategy, but round-1
(pre-reveal) hero decisions are solved with VectorCFR truncated at the
community-reveal boundary, leaf values from a PBS value evaluator (the
learned net, or an exact oracle). Round-2 decisions are solved exactly
with the Phase 1 deal-based resolver — the "river solved exactly"
pattern that Phase 3 scales to NLHE street boundaries.
"""

from __future__ import annotations

import numpy as np

from ..games.base import History, InfoSetKey
from ..games.leduc import LeducHoldem
from .pbs import (
    LEDUC_CARDS,
    LeducPBS,
    initial_pbs,
    representative_history,
    update_on_action,
    update_on_community,
)
from .resolve import resolve
from .vector_cfr import VectorCFR


def build_dl_resolving_strategy(
    base_game: LeducHoldem,
    blueprint_fn,
    hero_seat: int,
    leaf_evaluator,
    dl_iters: int = 200,
    round2_resolve_iters: int = 300,
    verbose: bool = False,
) -> dict[InfoSetKey, np.ndarray]:
    """Full behavioural strategy of the depth-limited re-solving agent.

    leaf_evaluator(comm, cont, y0, y1) -> (v0[6], v1[6]): PBS values at
    the reveal boundary (value net in production, exact oracle for
    validation). Interface identical to build_resolving_strategy
    otherwise; see that function for the range-carrying semantics.
    """
    strategy_out: dict[InfoSetKey, np.ndarray] = {}
    counts = {"dl": 0, "exact": 0}

    def hero_fn(history: History, player: int) -> np.ndarray:
        key = base_game.info_set_key(history, player)
        n = len(base_game.legal_actions(history))
        if key in strategy_out:
            return strategy_out[key][:n]
        return np.ones(n, dtype=np.float64) / n

    def walk(pbs: LeducPBS) -> None:
        rep = representative_history("J1", 0, pbs.community, pbs.actions)
        if base_game.is_terminal(rep):
            return

        _, _, r1_done = base_game._split_rounds(pbs.actions)
        if r1_done and pbs.community is None:
            for card in LEDUC_CARDS:
                walk(update_on_community(pbs, card))
            return

        player = base_game.current_player(rep)
        legal = base_game.legal_actions(rep)

        if player == hero_seat:
            if pbs.community is None:
                # Round 1: depth-limited vector CFR, net/oracle leaves.
                solver = VectorCFR(
                    base_game, pbs,
                    leaf_evaluator=leaf_evaluator,
                    iterations=dl_iters,
                )
                solver.solve()
                counts["dl"] += 1
                strategy_at = solver.strategy_at
            else:
                # Round 2: exact deal-based resolve (terminal-depth).
                result = resolve(base_game, pbs,
                                 iterations=round2_resolve_iters)
                counts["exact"] += 1
                strategy_at = result.strategy_at

            for card in LEDUC_CARDS:
                h = representative_history(
                    card, hero_seat, pbs.community, pbs.actions
                )
                key = base_game.info_set_key(h, hero_seat)
                if key not in strategy_out:
                    strategy_out[key] = strategy_at(h, hero_seat)
            update_fn = hero_fn
        else:
            update_fn = blueprint_fn

        for action in legal:
            child = update_on_action(pbs, base_game, player, action, update_fn)
            walk(child)

    walk(initial_pbs())

    if verbose:
        print(f"  hero_seat={hero_seat}: {counts['dl']} depth-limited + "
              f"{counts['exact']} exact resolves, "
              f"{len(strategy_out)} infosets")
    return strategy_out
