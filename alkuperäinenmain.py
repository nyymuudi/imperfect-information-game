#!/usr/bin/env python3
"""
Solve Kuhn Poker with vanilla CFR and verify against analytical Nash equilibrium.

Usage:
    python -m src.main [--iterations N]
"""

import argparse
import sys

from src.games.kuhn import KuhnPoker
from src.solvers.cfr import CFRSolver
from src.analysis.convergence import ConvergenceTracker, verify_kuhn_nash


def main():
    parser = argparse.ArgumentParser(description="CFR Solver for Kuhn Poker")
    parser.add_argument(
        "--iterations", "-n", type=int, default=10000,
        help="Number of CFR iterations (default: 10000)"
    )
    args = parser.parse_args()

    print(f"Solving Kuhn Poker with {args.iterations} CFR iterations...\n")

    game = KuhnPoker()
    solver = CFRSolver(game=game)
    tracker = ConvergenceTracker()

    strategy = solver.solve(
        iterations=args.iterations,
        callback=tracker.record,
        callback_freq=max(1, args.iterations // 20),
    )

    # ── Results ──
    print(tracker.summary())

    # ── Nash Verification ──
    print(f"\n{'=' * 50}")
    print("Nash Equilibrium Verification")
    print(f"{'=' * 50}")
    print(f"\nEquilibrium structure:\n{game.known_nash_description()}\n")
    print(f"Analytical game value (P0 EV): {game.known_game_value():.6f}")
    print(f"Final exploitability:          {solver.exploitability():.6f}\n")

    # Infer α from strategy
    j_bet = strategy.get("J:", [0, 0])
    alpha = j_bet[1] if hasattr(j_bet, '__len__') else 0
    print(f"Inferred α (bluff frequency): {alpha:.4f}")
    print(f"Expected K bet (3α):          {3 * alpha:.4f}")
    k_bet = strategy.get("K:", [0, 0])
    print(f"Actual K bet:                 {k_bet[1]:.4f}\n")

    results = verify_kuhn_nash(strategy, tolerance=0.03)
    all_match = True
    for key in sorted(results.keys()):
        r = results[key]
        status = "✓" if r["match"] else "✗"
        if not r["match"]:
            all_match = False
        print(f"  {status} {key}: {r['detail']}")

    print(f"\n{'All Nash structural properties verified!' if all_match else 'Some properties not yet converged — increase iterations.'}")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())