#!/usr/bin/env python3
"""
Phase 1 gate measurement: continual re-solving vs blueprint on Leduc.

Protocol (rebel_extension plan, Phase 1 exit criterion):
  1. Train a Deep CFR blueprint on Leduc (the strategy the search layer
     should improve on).
  2. Measure the blueprint's EXACT full-game exploitability.
  3. Build the continual re-solving agent's full behavioural strategy
     (unsafe re-solving, tabular CFR+ to terminal, ranges carried forward)
     for both seats.
  4. Measure the agent's EXACT full-game exploitability with the same
     machinery.
  Exit: agent exploitability <= blueprint exploitability.

Usage:
    python3 scripts/phase1_leduc_resolve.py
    python3 scripts/phase1_leduc_resolve.py --bp-iters 100 --resolve-iters 400
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.deep_cfr.state_encoder import LeducEncoder
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.search.resolve import build_resolving_strategy


def exact_exploitability_of(game, strategy_fn) -> float:
    """Exact full-game exploitability of a behavioural strategy.

    strategy_fn(history, player) -> probs over legal actions. Loads the
    strategy into a CFRSolver carrier (cumulative_strategy := probs) and
    uses its exact best-response machinery — same method as
    validate_deepcfr_leduc.py, so numbers are directly comparable.
    """
    ref = CFRSolver(game=game, linear_averaging=True)

    def walk(history):
        if game.is_terminal(history):
            return
        player = game.current_player(history)
        acts = game.legal_actions(history)
        key = game.info_set_key(history, player)
        if key not in ref.info_sets:
            probs = np.asarray(strategy_fn(history, player), dtype=np.float64)
            data = ref._get_or_create_info_set(key, acts)
            data.cumulative_strategy = probs[: len(acts)].copy()
        for a in acts:
            walk(game.apply_action(history, a))

    for init_h, _ in game.initial_histories():
        walk(init_h)
    return ref.exploitability()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp-iters", type=int, default=100,
                    help="Deep CFR blueprint training iterations")
    ap.add_argument("--traversals", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--resolve-iters", type=int, default=400,
                    help="tabular CFR+ iterations per re-solve")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    game = LeducHoldem()
    encoder = LeducEncoder()

    # ── 1. Blueprint: Deep CFR on Leduc ─────────────────────────────────────
    print(f"[1/4] Training Deep CFR blueprint "
          f"({args.bp_iters} iters × {args.traversals} traversals)...")
    t0 = time.time()
    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,
        buffer_capacity=500_000,
        hidden_size=args.hidden,
        train_epochs=args.epochs,
        traversals_per_iter=args.traversals,
        use_cpp_engine=False,
        device="cpu",
        lr=1e-3,
    )
    solver.solve(iterations=args.bp_iters)
    print(f"      done in {time.time()-t0:.1f}s")

    def blueprint_fn(history, player):
        state = encoder.encode(history, player)
        n = len(game.legal_actions(history))
        return solver._get_regret_strategy(state, n)

    # ── 2. Blueprint exploitability (exact) ─────────────────────────────────
    print("[2/4] Blueprint exact exploitability...")
    bp_expl = exact_exploitability_of(game, blueprint_fn)
    print(f"      blueprint_exploitability = {bp_expl:.5f}")

    # ── 3. Continual re-solving agent, both seats ───────────────────────────
    print(f"[3/4] Building re-solving agent "
          f"(resolve_iters={args.resolve_iters})...")
    t0 = time.time()
    agent_strategy: dict = {}
    for seat in (0, 1):
        seat_strat = build_resolving_strategy(
            game, blueprint_fn, hero_seat=seat,
            resolve_iters=args.resolve_iters, verbose=True,
        )
        agent_strategy.update(seat_strat)
    print(f"      done in {time.time()-t0:.1f}s "
          f"({len(agent_strategy)} infosets)")

    def agent_fn(history, player):
        key = game.info_set_key(history, player)
        n = len(game.legal_actions(history))
        if key in agent_strategy:
            return agent_strategy[key][:n]
        return np.ones(n) / n

    # ── 4. Agent exploitability (exact, same machinery) ─────────────────────
    print("[4/4] Re-solving agent exact exploitability...")
    ag_expl = exact_exploitability_of(game, agent_fn)

    print("\n== Phase 1 gate ==")
    print(f"blueprint exploitability : {bp_expl:.5f}")
    print(f"re-solving exploitability: {ag_expl:.5f}")
    delta = ag_expl - bp_expl
    verdict = "PASS (agent <= blueprint)" if delta <= 0 else "FAIL"
    print(f"delta                    : {delta:+.5f}  -> {verdict}")
    print("(tabular reference floor ~0.138 @1000 tabular iters)")
    return 0 if delta <= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
