#!/usr/bin/env python3
"""
Solve preflop heads-up NLHE with abstraction and CFR.

Usage:
    python -m src.nlhe_main [--buckets K] [--iterations N] [--stack S]
"""

import argparse
import time
import sys

from src.abstraction.card_abstraction import CardAbstraction
from src.games.nlhe_preflop import PreflopNLHE
from src.solvers.cfr import CFRSolver


def main():
    parser = argparse.ArgumentParser(description="Preflop NLHE CFR Solver")
    parser.add_argument("--buckets", "-k", type=int, default=8)
    parser.add_argument("--iterations", "-n", type=int, default=2000)
    parser.add_argument("--stack", "-s", type=float, default=100.0)
    parser.add_argument("--equity-sims", type=int, default=500)
    args = parser.parse_args()

    # ── Step 1: Card Abstraction ──
    print(f"Building {args.buckets}-bucket card abstraction...")
    t0 = time.time()
    abstraction = CardAbstraction.from_equity(
        num_buckets=args.buckets,
        num_simulations=args.equity_sims,
        seed=42,
    )
    print(f"  Done in {time.time()-t0:.1f}s\n")
    print(abstraction.summary())

    # ── Step 2: Game ──
    game = PreflopNLHE(
        abstraction=abstraction,
        stack_bb=args.stack,
    )
    print(f"\nGame: HU Preflop NLHE, {args.stack:.0f}BB effective")
    print(f"Bucket pairs: {len(game.initial_histories())}")
    print(f"Raise schedule: {game.raise_sizes} BB\n")

    # ── Step 3: CFR ──
    print(f"Running {args.iterations} CFR iterations...")
    solver = CFRSolver(game=game, linear_averaging=True)
    t0 = time.time()

    def callback(s, i):
        elapsed = time.time() - t0
        print(f"  iter {i:>5d}: info_sets={len(s.info_sets)}, time={elapsed:.1f}s")

    strategy = solver.solve(
        iterations=args.iterations,
        callback=callback,
        callback_freq=max(1, args.iterations // 10),
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s | {len(strategy)} info sets\n")

    # ── Step 4: Strategy Output ──
    print("=" * 60)
    print("SB OPENING STRATEGY (P0 first action)")
    print("=" * 60)
    for b in range(args.buckets):
        key = f"B{b}|"
        if key in strategy:
            s = strategy[key]
            actions = game.legal_actions((b, 0))
            parts = [f"{a}={s[i]:.0%}" for i, a in enumerate(actions)]
            name = game.bucket_name(b)
            print(f"  {name:>45s}: {'  '.join(parts)}")

    print(f"\n{'=' * 60}")
    print("BB RESPONSE TO SB RAISE (P1 after 'r')")
    print("=" * 60)
    for b in range(args.buckets):
        key = f"B{b}|r"
        if key in strategy:
            s = strategy[key]
            actions = game.legal_actions((0, b, "r"))
            parts = [f"{a}={s[i]:.0%}" for i, a in enumerate(actions)]
            name = game.bucket_name(b)
            print(f"  {name:>45s}: {'  '.join(parts)}")

    print(f"\n{'=' * 60}")
    print("BB RESPONSE TO SB LIMP (P1 after 'c')")
    print("=" * 60)
    for b in range(args.buckets):
        key = f"B{b}|c"
        if key in strategy:
            s = strategy[key]
            actions = game.legal_actions((0, b, "c"))
            parts = [f"{a}={s[i]:.0%}" for i, a in enumerate(actions)]
            name = game.bucket_name(b)
            print(f"  {name:>45s}: {'  '.join(parts)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())