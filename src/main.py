#!/usr/bin/env python3
"""
Domain-agnostic CFR solver for Kuhn Poker and Leduc Hold'em.

Usage:
    python -m src.main --game kuhn --iterations 10000
    python -m src.main --game leduc --iterations 5000 --cfr_mode linear
"""

import argparse
import sys
import numpy as np

from src.games.kuhn import KuhnPoker
from src.games.leduc import LeducHoldem
from src.solvers.cfr import CFRSolver
from src.analysis.convergence import ConvergenceTracker, verify_kuhn_nash


# ═══════════════════════════════════════════════════════════════════
# Best response & exploitability (game‑agnostic)
# ═══════════════════════════════════════════════════════════════════

def best_response_value(game, player, opponent_strategy):
    """
    Laske pelaajan `player` paras vastaus kiinteälle vastustajan strategialle.
    opponent_strategy: dict info_set_key → np.ndarray (todennäköisyydet
    samassa järjestyksessä kuin game.legal_actions).
    """
    def traverse(history, reach_opp, acting_player):
        if game.is_terminal(history):
            return game.terminal_payoffs(history)[player]

        if acting_player == player:
            best = -float('inf')
            for a in game.legal_actions(history):
                next_h = game.apply_action(history, a)
                next_player = game.current_player(next_h)
                val = traverse(next_h, reach_opp, next_player)
                if val > best:
                    best = val
            return best
        else:
            info = game.info_set_key(history, acting_player)
            strat = opponent_strategy.get(info)
            if strat is None:
                actions = game.legal_actions(history)
                strat = np.ones(len(actions)) / len(actions)
            ev = 0.0
            for i, prob in enumerate(strat):
                if prob > 0:
                    next_h = game.apply_action(history, game.legal_actions(history)[i])
                    next_player = game.current_player(next_h)
                    ev += prob * traverse(next_h, reach_opp * prob, next_player)
            return ev

    total = 0.0
    for hist, prob in game.initial_histories():
        total += prob * traverse(hist, 1.0, game.current_player(hist))
    return total


def exploitability(game, avg_strategies):
    """
    Palauta annetun strategiaprofiilin exploitability.
    avg_strategies: lista [P0_dict, P1_dict], kukin dict info_set → np.array
    """
    br0 = best_response_value(game, 0, avg_strategies[1])
    br1 = best_response_value(game, 1, avg_strategies[0])
    return br0 + br1


# ═══════════════════════════════════════════════════════════════════
# Pääohjelma
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CFR Solver for Kuhn / Leduc")
    parser.add_argument(
        "--game", choices=["kuhn", "leduc"], default="kuhn",
        help="Game to solve (default: kuhn)"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10000,
        help="Number of CFR iterations (default: 10000)"
    )
    parser.add_argument(
        "--cfr_mode", choices=["vanilla", "linear"], default="linear",
        help="CFR averaging scheme (default: linear)"
    )
    args = parser.parse_args()

    if args.game == "kuhn":
        game = KuhnPoker()
    elif args.game == "leduc":
        game = LeducHoldem()

    print(f"Solving {args.game.upper()} with {args.cfr_mode} CFR ({args.iterations} iterations)...\n")

    solver = CFRSolver(game=game)
    tracker = ConvergenceTracker()

    strategy_list = solver.solve(
        iterations=args.iterations,
        callback=lambda solver, it: print(f"Iteration {it}/{args.iterations}"),
        callback_freq=max(1, args.iterations // 20),
        mode=args.cfr_mode
    )
    # strategy_list = [p0_dict, p1_dict]; arvot ovat np.ndarray

    # ── Tulokset ──
    print(tracker.summary())

    if args.game == "kuhn":
        # Yhdistetään yhdeksi sanakirjaksi Kuhn‑verifiointia varten
        combined = {**strategy_list[0], **strategy_list[1]}

        print(f"\n{'=' * 50}")
        print("Nash Equilibrium Verification (Kuhn)")
        print(f"{'=' * 50}")
        print(f"\nEquilibrium structure:\n{game.known_nash_description()}\n")
        print(f"Analytical game value (P0 EV): {game.known_game_value():.6f}")
        print(f"Final exploitability:          {solver.exploitability():.6f}\n")

        j_bet = combined["J:"][1]
        alpha = j_bet
        print(f"Inferred α (bluff frequency): {alpha:.4f}")
        if alpha > 0:
            print(f"Expected K bet (3α):          {3 * alpha:.4f}")
            print(f"Actual K bet:                 {combined['K:'][1]:.4f}\n")

        results = verify_kuhn_nash(combined, tolerance=0.03)
        all_match = True
        for key in sorted(results.keys()):
            r = results[key]
            status = "✓" if r["match"] else "✗"
            if not r["match"]:
                all_match = False
            print(f"  {status} {key}: {r['detail']}")

        print(f"\n{'All Nash structural properties verified!' if all_match else 'Some properties not yet converged — increase iterations.'}")
        return 0 if all_match else 1

    else:  # leduc
        p0_strat, p1_strat = strategy_list
        expl = exploitability(game, [p0_strat, p1_strat])
        print(f"\nFinal exploitability (Leduc): {expl:.6f}")
        print(f"Info sets solved: {len(p0_strat) + len(p1_strat)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())