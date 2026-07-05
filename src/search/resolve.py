"""
Continual re-solving on Leduc with an exact tabular subgame solver.

Phase 1 of the ReBeL extension plan: no neural nets, no depth limit —
each subgame is solved to terminal with the existing tabular CFR
(src/solvers/cfr.py, cfr_plus=True). Unsafe re-solving (Burch et al. 2014
terminology): the subgame root distribution comes straight from the PBS
ranges with no opponent CBV constraints. The safe gadget is added only if
unsafe re-solving measurably raises full-game exploitability.

Components:

    LeducSubgame     ExtensiveFormGame over deals drawn from a PBS: the
                     cross-product of both ranges (× community candidates
                     pre-reveal), card-removal filtered, prefix actions
                     pre-applied. Info-set keys delegate to LeducHoldem —
                     all deals share the public prefix, so keys are
                     consistent across deals without local re-keying.

    resolve()        Solve the subgame rooted at a PBS; returns per-rank
                     root strategies for the acting player.

    build_resolving_strategy()
                     Walk the public tree; at every hero decision node
                     re-solve and record the root strategy; carry hero's
                     range forward with the strategies actually recorded
                     (continual re-solving) and the opponent's range with
                     the blueprint. Returns a full behavioural strategy
                     {info_set_key: probs} for the hero seat, suitable for
                     exact best-response evaluation.
"""

from __future__ import annotations

import numpy as np

from ..games.base import ExtensiveFormGame, History, Action, InfoSetKey
from ..games.leduc import LeducHoldem
from ..solvers.cfr import CFRSolver
from .pbs import (
    LEDUC_CARDS,
    LeducPBS,
    representative_history,
    update_on_action,
    update_on_community,
)


# ── Subgame over PBS deals ───────────────────────────────────────────────────

class LeducSubgame(ExtensiveFormGame):
    """LeducHoldem restricted to the subtree rooted at a PBS.

    initial_histories(): weighted concrete deals (c0, c1, comm, *prefix).
    Weights = range0[c0] · range1[c1] · P(comm), renormalised after
    card-removal filtering. Pre-reveal, comm is uniform over the four
    cards outside (c0, c1); post-reveal it is fixed by the PBS.
    """

    def __init__(self, base_game: LeducHoldem, pbs: LeducPBS):
        self.base_game = base_game
        self.pbs = pbs
        self._initial = self._build_initial()

    def _build_initial(self) -> list[tuple[History, float]]:
        r0 = self.pbs.range_array(0)
        r1 = self.pbs.range_array(1)
        prefix = self.pbs.actions
        comm_fixed = self.pbs.community

        deals: list[tuple[History, float]] = []
        for i, c0 in enumerate(LEDUC_CARDS):
            if r0[i] <= 0.0:
                continue
            for j, c1 in enumerate(LEDUC_CARDS):
                if i == j or r1[j] <= 0.0:
                    continue
                pair_w = r0[i] * r1[j]
                if comm_fixed is not None:
                    if comm_fixed in (c0, c1):
                        continue
                    deals.append(((c0, c1, comm_fixed) + prefix, pair_w))
                else:
                    remaining = [c for c in LEDUC_CARDS if c not in (c0, c1)]
                    w = pair_w / len(remaining)
                    for comm in remaining:
                        deals.append(((c0, c1, comm) + prefix, w))

        total = sum(w for _, w in deals)
        if total <= 0.0:
            return []
        return [(h, w / total) for h, w in deals]

    # ── Delegation to the base game ──────────────────────────────────────────

    def num_players(self) -> int:
        return 2

    def initial_histories(self) -> list[tuple[History, float]]:
        return self._initial

    def is_terminal(self, history: History) -> bool:
        return self.base_game.is_terminal(history)

    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        return self.base_game.terminal_payoffs(history)

    def current_player(self, history: History) -> int:
        return self.base_game.current_player(history)

    def legal_actions(self, history: History) -> list[Action]:
        return self.base_game.legal_actions(history)

    def apply_action(self, history: History, action: Action) -> History:
        return self.base_game.apply_action(history, action)

    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        return self.base_game.info_set_key(history, player)


# ── Re-solve one PBS ─────────────────────────────────────────────────────────

class ResolveResult:
    """Solved subgame: average strategies + query helpers."""

    def __init__(self, solver: CFRSolver, subgame: LeducSubgame):
        self.solver = solver
        self.subgame = subgame
        self.strategies: dict[InfoSetKey, np.ndarray] = {
            key: data.average_strategy()
            for key, data in solver.info_sets.items()
        }

    def strategy_at(self, history: History, player: int) -> np.ndarray:
        """Resolved probs over legal actions; uniform for unseen infosets."""
        key = self.subgame.info_set_key(history, player)
        n = len(self.subgame.legal_actions(history))
        if key in self.strategies:
            return np.asarray(self.strategies[key][:n], dtype=np.float64)
        return np.ones(n, dtype=np.float64) / n


def resolve(
    base_game: LeducHoldem,
    pbs: LeducPBS,
    iterations: int = 400,
) -> ResolveResult:
    """Solve the subgame rooted at `pbs` to terminal with tabular CFR+."""
    subgame = LeducSubgame(base_game, pbs)
    solver = CFRSolver(game=subgame, linear_averaging=True, cfr_plus=True)
    if subgame.initial_histories():
        solver.solve(iterations=iterations)
    return ResolveResult(solver, subgame)


# ── Continual re-solving over the public tree ────────────────────────────────

def build_resolving_strategy(
    base_game: LeducHoldem,
    blueprint_fn,
    hero_seat: int,
    resolve_iters: int = 400,
    verbose: bool = False,
) -> dict[InfoSetKey, np.ndarray]:
    """Full behavioural strategy of the continual re-solving agent.

    blueprint_fn(history, player) -> probs over legal actions: the model
    of the OPPONENT (and the belief model the opponent has of us before
    we deviate — hero's own range is updated with the strategies this
    walk actually records, not the blueprint).

    Returns {info_set_key: probs} covering every hero infoset in the
    public tree, including zero-reach ones (uniform-fallback ranges), so
    an exact best response is fully defined.
    """
    strategy_out: dict[InfoSetKey, np.ndarray] = {}
    n_resolves = 0

    def hero_strategy_fn_factory(recorded: dict[InfoSetKey, np.ndarray]):
        def fn(history: History, player: int) -> np.ndarray:
            key = base_game.info_set_key(history, player)
            n = len(base_game.legal_actions(history))
            if key in recorded:
                return recorded[key][:n]
            return np.ones(n, dtype=np.float64) / n
        return fn

    hero_fn = hero_strategy_fn_factory(strategy_out)

    def walk(pbs: LeducPBS) -> None:
        nonlocal n_resolves
        rep = representative_history("J1", 0, pbs.community, pbs.actions)
        if base_game.is_terminal(rep):
            return

        # Community reveal boundary: round 1 done, card not yet public.
        _, _, r1_done = base_game._split_rounds(pbs.actions)
        if r1_done and pbs.community is None:
            for card in LEDUC_CARDS:
                walk(update_on_community(pbs, card))
            return

        player = base_game.current_player(rep)
        legal = base_game.legal_actions(rep)

        if player == hero_seat:
            result = resolve(base_game, pbs, iterations=resolve_iters)
            n_resolves += 1
            if verbose and n_resolves % 20 == 0:
                print(f"  [resolve {n_resolves}] actions={pbs.actions} "
                      f"comm={pbs.community}")
            # Record the ROOT strategy for each rank (J/Q/K share an
            # infoset — Leduc keys are rank-based).
            for card in LEDUC_CARDS:
                h = representative_history(
                    card, hero_seat, pbs.community, pbs.actions
                )
                key = base_game.info_set_key(h, hero_seat)
                if key not in strategy_out:
                    strategy_out[key] = result.strategy_at(h, hero_seat)
            update_fn = hero_fn
        else:
            update_fn = blueprint_fn

        for action in legal:
            child = update_on_action(pbs, base_game, player, action, update_fn)
            walk(child)

    from .pbs import initial_pbs
    walk(initial_pbs())

    if verbose:
        print(f"  hero_seat={hero_seat}: {n_resolves} resolves, "
              f"{len(strategy_out)} infosets recorded")
    return strategy_out
