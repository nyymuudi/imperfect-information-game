"""
Convergence analysis tools for CFR solvers.

Provides metrics to measure how close the computed strategy is to
a Nash equilibrium, and tools to track convergence over iterations.
"""

import numpy as np
from dataclasses import dataclass
from ..solvers.cfr import CFRSolver


@dataclass
class ConvergenceSnapshot:
    """Single observation of solver state at a given iteration."""
    iteration: int
    exploitability: float
    strategy: dict[str, np.ndarray]


class ConvergenceTracker:
    """
    Records exploitability and strategy snapshots during CFR solving.
    
    Usage:
        tracker = ConvergenceTracker()
        solver.solve(iterations=10000, callback=tracker.record, callback_freq=100)
        tracker.summary()
    """

    def __init__(self):
        self.snapshots: list[ConvergenceSnapshot] = []

    def record(self, solver: CFRSolver, iteration: int) -> None:
        """Callback for CFRSolver.solve()."""
        exploit = solver.exploitability()
        strat = {
            k: v.average_strategy().copy()
            for k, v in solver.info_sets.items()
        }
        self.snapshots.append(ConvergenceSnapshot(iteration, exploit, strat))

    def iterations(self) -> np.ndarray:
        return np.array([s.iteration for s in self.snapshots])

    def exploitabilities(self) -> np.ndarray:
        return np.array([s.exploitability for s in self.snapshots])

    def summary(self) -> str:
        """Human-readable convergence summary."""
        if not self.snapshots:
            return "No data recorded."

        final = self.snapshots[-1]
        lines = [
            f"CFR Convergence Summary",
            f"{'=' * 50}",
            f"Iterations:        {final.iteration}",
            f"Final exploit.:    {final.exploitability:.6f}",
            f"Info sets:         {len(final.strategy)}",
            f"",
            f"Convergence rate (exploit. at checkpoints):",
        ]

        checkpoints = [self.snapshots[0]] + self.snapshots[::max(1, len(self.snapshots) // 5)] + [final]
        seen = set()
        for s in checkpoints:
            if s.iteration not in seen:
                seen.add(s.iteration)
                lines.append(f"  iter {s.iteration:>6d}: {s.exploitability:.6f}")

        lines.append(f"\nFinal average strategy:")
        for key in sorted(final.strategy.keys()):
            strat = final.strategy[key]
            actions_str = ", ".join(f"{v:.4f}" for v in strat)
            lines.append(f"  {key:>10s}: [{actions_str}]")

        return "\n".join(lines)


def verify_kuhn_nash(
    strategy: dict[str, np.ndarray],
    tolerance: float = 0.02,
) -> dict[str, dict]:
    """
    Verify CFR solution against structural Nash equilibrium properties
    of Kuhn Poker that hold for ALL α in [0, 1/3].
    
    The Nash equilibrium family is parameterized by α ∈ [0, 1/3]:
        P0: J bets α,  K bets 3α,  Q always checks
        P0 facing check-bet: K calls, J and Q fold
        P1 facing bet: K calls, J folds, Q calls 1/3
        P1 after check: K always bets, J never bets
    
    Structural invariants (valid for any α):
        1. K bet frequency = 3 × J bet frequency  (β = 3α)
        2. Q never bets (P0 always checks with Q)
        3. P1 always calls with K, always folds with J (both lines)
        4. P1 calls with Q facing bet ≈ 1/3
        5. P0 facing check-bet: calls with K, folds with J
    
    Returns dict of property_name → {expected, actual, match, detail}.
    """
    results = {}

    # Property 1: β = 3α (K bet / J bet ratio)
    j_bet = strategy.get("J:", np.array([0.5, 0.5]))[1]  # J bet prob
    k_bet = strategy.get("K:", np.array([0.5, 0.5]))[1]  # K bet prob
    expected_k_bet = 3 * j_bet
    ratio_error = abs(k_bet - expected_k_bet)
    results["β=3α"] = {
        "detail": f"J bets {j_bet:.4f}, K bets {k_bet:.4f}, expected K={expected_k_bet:.4f}",
        "match": ratio_error < tolerance or j_bet < 0.01,  # If α≈0, ratio is undefined
        "error": ratio_error,
    }

    # Property 2: Q never bets initially
    q_check = strategy.get("Q:", np.array([0.5, 0.5]))[0]
    results["Q always checks"] = {
        "detail": f"Q check prob = {q_check:.4f}",
        "match": q_check > 1.0 - tolerance,
        "error": 1.0 - q_check,
    }

    # Property 3: P1 always calls with K facing bet
    k_call = strategy.get("K:b", np.array([0.5, 0.5]))[1]
    results["P1 K calls bet"] = {
        "detail": f"K:b call = {k_call:.4f}",
        "match": k_call > 1.0 - tolerance,
        "error": 1.0 - k_call,
    }

    # Property 4: P1 always folds with J facing bet
    j_fold = strategy.get("J:b", np.array([0.5, 0.5]))[0]
    results["P1 J folds to bet"] = {
        "detail": f"J:b fold = {j_fold:.4f}",
        "match": j_fold > 1.0 - tolerance,
        "error": 1.0 - j_fold,
    }

    # Property 5: P1 calls with Q facing bet ≈ 1/3
    q_call_bet = strategy.get("Q:b", np.array([0.5, 0.5]))[1]
    results["P1 Q calls 1/3"] = {
        "detail": f"Q:b call = {q_call_bet:.4f}, expected ≈ 0.333",
        "match": abs(q_call_bet - 1/3) < tolerance,
        "error": abs(q_call_bet - 1/3),
    }

    # Property 6: P1 always bets with K after check
    k_bet_after_check = strategy.get("K:c", np.array([0.5, 0.5]))[1]
    results["P1 K bets after check"] = {
        "detail": f"K:c bet = {k_bet_after_check:.4f}",
        "match": k_bet_after_check > 1.0 - tolerance,
        "error": 1.0 - k_bet_after_check,
    }

    # Property 7: P1 bets with J after check ≈ 1/3 (not never!)
    # This makes P0 with Q indifferent at check-bet node
    j_bet_after_check = strategy.get("J:c", np.array([0.5, 0.5]))[1]
    results["P1 J bets 1/3 after check"] = {
        "detail": f"J:c bet = {j_bet_after_check:.4f}, expected ≈ 0.333",
        "match": abs(j_bet_after_check - 1/3) < tolerance,
        "error": abs(j_bet_after_check - 1/3),
    }

    # Property 8: P1 never bets with Q after check
    q_check_after_check = strategy.get("Q:c", np.array([0.5, 0.5]))[0]
    results["P1 Q checks after check"] = {
        "detail": f"Q:c check = {q_check_after_check:.4f}",
        "match": q_check_after_check > 1.0 - tolerance,
        "error": 1.0 - q_check_after_check,
    }

    # Property 9: P0 calls with K facing check-bet
    k_call_cb = strategy.get("K:cb", np.array([0.5, 0.5]))[1]
    results["P0 K calls check-bet"] = {
        "detail": f"K:cb call = {k_call_cb:.4f}",
        "match": k_call_cb > 1.0 - tolerance,
        "error": 1.0 - k_call_cb,
    }

    # Property 10: P0 folds with J facing check-bet
    j_fold_cb = strategy.get("J:cb", np.array([0.5, 0.5]))[0]
    results["P0 J folds check-bet"] = {
        "detail": f"J:cb fold = {j_fold_cb:.4f}",
        "match": j_fold_cb > 1.0 - tolerance,
        "error": 1.0 - j_fold_cb,
    }

    # Property 11: α ∈ [0, 1/3]
    results["α in valid range"] = {
        "detail": f"α (J bet prob) = {j_bet:.4f}",
        "match": -tolerance < j_bet < 1/3 + tolerance,
        "error": max(0, j_bet - 1/3, -j_bet),
    }

    return results