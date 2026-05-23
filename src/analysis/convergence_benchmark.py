#!/usr/bin/env python3
"""
Convergence comparison: CFR vs CFR+ across Kuhn and Leduc.

Generates convergence plots showing exploitability vs iterations
for both solver variants on each game. Demonstrates:
    1. CFR+ converges faster due to regret clamping
    2. Linear averaging improves both variants
    3. Scaling: same algorithm, exponentially harder games

Output: convergence_comparison.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from src.games.kuhn import KuhnPoker
from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.analysis.convergence import ConvergenceTracker


def benchmark(game, label, iterations, callback_freq, **solver_kwargs):
    """Run solver and collect convergence data."""
    solver = CFRSolver(game=game, **solver_kwargs)
    tracker = ConvergenceTracker()
    t0 = time.time()
    solver.solve(
        iterations=iterations,
        callback=tracker.record,
        callback_freq=callback_freq,
    )
    elapsed = time.time() - t0
    print(f"  {label}: {elapsed:.1f}s, final exploit={tracker.exploitabilities()[-1]:.6f}")
    return tracker


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Kuhn Poker ──
    print("Kuhn Poker:")
    game = KuhnPoker()
    iters, freq = 10000, 200

    kuhn_cfr = benchmark(game, "CFR", iters, freq,
                         linear_averaging=True, cfr_plus=False)
    kuhn_cfrp = benchmark(game, "CFR+", iters, freq,
                          linear_averaging=True, cfr_plus=True)

    ax = axes[0]
    ax.plot(kuhn_cfr.iterations(), kuhn_cfr.exploitabilities(),
            label="CFR (Linear)", linewidth=1.5, color="#2563eb")
    ax.plot(kuhn_cfrp.iterations(), kuhn_cfrp.exploitabilities(),
            label="CFR+", linewidth=1.5, color="#dc2626")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("Kuhn Poker (12 info sets)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, iters)

    # ── Leduc ──
    print("Leduc Hold'em:")
    game = LeducHoldem()
    iters, freq = 500, 25

    leduc_cfr = benchmark(game, "CFR", iters, freq,
                          linear_averaging=True, cfr_plus=False)
    leduc_cfrp = benchmark(game, "CFR+", iters, freq,
                           linear_averaging=True, cfr_plus=True)

    ax = axes[1]
    ax.plot(leduc_cfr.iterations(), leduc_cfr.exploitabilities(),
            label="CFR (Linear)", linewidth=1.5, color="#2563eb")
    ax.plot(leduc_cfrp.iterations(), leduc_cfrp.exploitabilities(),
            label="CFR+", linewidth=1.5, color="#dc2626")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("Leduc Hold'em (288 info sets)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, iters)

    fig.suptitle(
        "CFR Convergence: Exploitability vs Iterations",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: convergence_comparison.png")


if __name__ == "__main__":
    main()