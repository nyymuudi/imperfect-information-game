#!/usr/bin/env python3
"""
Convergence comparison: CFR vs CFR+ vs MCCFR across Kuhn and Leduc.

Generates a 2×2 plot:
    Top row:    Kuhn Poker — exploitability vs iterations, exploitability vs time
    Bottom row: Leduc Hold'em — same metrics

Demonstrates:
    1. MCCFR iterations are much faster but noisier
    2. CFR+ converges faster per iteration than vanilla CFR
    3. Wall-clock comparison reveals MCCFR's true advantage on larger games

Output: convergence_comparison.png
"""

import time
import matplotlib.pyplot as plt
from src.games.kuhn import KuhnPoker
from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.solvers.mccfr import MCCFRSolver


class TimedTracker:
    """Records exploitability with both iteration count and wall-clock time."""
    def __init__(self, start_time: float):
        self.t0 = start_time
        self.iters = []
        self.exploits = []
        self.times = []

    def record(self, solver, iteration):
        e = solver.exploitability()
        self.iters.append(iteration)
        self.exploits.append(e)
        self.times.append(time.time() - self.t0)


def run_cfr(game, label, iterations, freq, **kwargs):
    solver = CFRSolver(game=game, **kwargs)
    tracker = TimedTracker(time.time())
    solver.solve(iterations=iterations, callback=tracker.record, callback_freq=freq)
    print(f"  {label:>20s}: {time.time()-tracker.t0:.1f}s, exploit={tracker.exploits[-1]:.5f}")
    return tracker


def run_mccfr(game, label, iterations, freq, seed=42):
    solver = MCCFRSolver(game=game, linear_averaging=True, cfr_plus=True)
    tracker = TimedTracker(time.time())
    solver.solve(iterations=iterations, callback=tracker.record, callback_freq=freq, seed=seed)
    print(f"  {label:>20s}: {time.time()-tracker.t0:.1f}s, exploit={tracker.exploits[-1]:.5f}")
    return tracker


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ═══ Kuhn Poker ═══
    print("Kuhn Poker:")
    game = KuhnPoker()

    cfr_k  = run_cfr(game, "CFR (Linear)", 10000, 200,
                      linear_averaging=True, cfr_plus=False)
    cfrp_k = run_cfr(game, "CFR+", 10000, 200,
                      linear_averaging=True, cfr_plus=True)
    mccfr_k = run_mccfr(game, "MCCFR", 100000, 2000)

    # Kuhn: exploitability vs iterations
    ax = axes[0][0]
    ax.plot(cfr_k.iters, cfr_k.exploits, label="CFR", lw=1.5, color="#2563eb")
    ax.plot(cfrp_k.iters, cfrp_k.exploits, label="CFR+", lw=1.5, color="#f59e0b")
    ax.plot(mccfr_k.iters, mccfr_k.exploits, label="MCCFR", lw=1.2, color="#dc2626", alpha=0.8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("Kuhn Poker — vs Iterations")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Kuhn: exploitability vs wall-clock time
    ax = axes[0][1]
    ax.plot(cfr_k.times, cfr_k.exploits, label="CFR", lw=1.5, color="#2563eb")
    ax.plot(cfrp_k.times, cfrp_k.exploits, label="CFR+", lw=1.5, color="#f59e0b")
    ax.plot(mccfr_k.times, mccfr_k.exploits, label="MCCFR", lw=1.2, color="#dc2626", alpha=0.8)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Exploitability")
    ax.set_title("Kuhn Poker — vs Time")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ═══ Leduc Hold'em ═══
    print("Leduc Hold'em:")
    game = LeducHoldem()

    cfr_l  = run_cfr(game, "CFR (Linear)", 300, 15,
                      linear_averaging=True, cfr_plus=False)
    cfrp_l = run_cfr(game, "CFR+", 300, 15,
                      linear_averaging=True, cfr_plus=True)
    mccfr_l = run_mccfr(game, "MCCFR", 100000, 5000)

    # Filter positive exploitabilities for log scale
    def filter_pos(iters, exploits):
        mask = [e > 0 for e in exploits]
        return [i for i, m in zip(iters, mask) if m], [e for e, m in zip(exploits, mask) if m]

    # Leduc: exploitability vs iterations
    ax = axes[1][0]
    i, e = filter_pos(cfr_l.iters, cfr_l.exploits)
    if i: ax.plot(i, e, label="CFR", lw=1.5, color="#2563eb")
    i, e = filter_pos(cfrp_l.iters, cfrp_l.exploits)
    if i: ax.plot(i, e, label="CFR+", lw=1.5, color="#f59e0b")
    i, e = filter_pos(mccfr_l.iters, mccfr_l.exploits)
    if i: ax.plot(i, e, label="MCCFR", lw=1.2, color="#dc2626", alpha=0.8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("Leduc Hold'em — vs Iterations")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Leduc: exploitability vs time
    ax = axes[1][1]
    i, e = filter_pos(cfr_l.times, cfr_l.exploits)
    if i: ax.plot(i, e, label="CFR", lw=1.5, color="#2563eb")
    i, e = filter_pos(cfrp_l.times, cfrp_l.exploits)
    if i: ax.plot(i, e, label="CFR+", lw=1.5, color="#f59e0b")
    i, e = filter_pos(mccfr_l.times, mccfr_l.exploits)
    if i: ax.plot(i, e, label="MCCFR", lw=1.2, color="#dc2626", alpha=0.8)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Exploitability")
    ax.set_title("Leduc Hold'em — vs Time")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CFR Solver Variants: Convergence Analysis",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: convergence_comparison.png")


if __name__ == "__main__":
    main()