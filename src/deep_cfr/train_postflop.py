#!/usr/bin/env python3
"""
Train Deep CFR on Postflop Heads-Up NLHE.

Usage:
    python -m src.deep_cfr.train_postflop [--iterations N] [--traversals T]
"""

import argparse
import time
import torch
import numpy as np
from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.deep_cfr.state_encoder import NLHEEncoder
from src.abstraction.equity import card_to_str


def main():
    parser = argparse.ArgumentParser(description="Deep CFR on Postflop NLHE")
    parser.add_argument("--iterations", "-n", type=int, default=50)
    parser.add_argument("--traversals", "-t", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--buffer", type=int, default=200000)
    parser.add_argument("--stack", type=float, default=200.0)
    args = parser.parse_args()

    game = PostflopNLHE(starting_stack=args.stack, max_raises_per_street=2)
    encoder = NLHEEncoder(starting_stack=args.stack)

    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,
        buffer_capacity=args.buffer,
        hidden_size=args.hidden,
        train_epochs=30,
        train_batch=256,
        traversals_per_iter=args.traversals,
        lr=1e-3,
    )

    print(f"Deep CFR on HU Postflop NLHE")
    print(f"  Stack: {args.stack:.0f}BB | Iterations: {args.iterations}")
    print(f"  Traversals/iter: {args.traversals} | Hidden: {args.hidden}")
    print(f"  Buffer capacity: {args.buffer:,}\n")

    t0 = time.time()

    def cb(s, i):
        elapsed = time.time() - t0
        rate = i * args.traversals * 2 / elapsed
        print(f"  iter {i:>4d}: MR={len(s.regret_buffer):>7,}, "
              f"MΠ={len(s.strategy_buffer):>7,}, "
              f"time={elapsed:>6.1f}s, {rate:.0f} trav/s")

    strategy_net = solver.solve(
        iterations=args.iterations,
        callback=cb,
        callback_freq=max(1, args.iterations // 20),
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Samples: MR={len(solver.regret_buffer):,}, MΠ={len(solver.strategy_buffer):,}\n")

    # ── Sample strategies ──
    rng = np.random.default_rng(42)
    print("=" * 60)
    print("PREFLOP OPENING STRATEGIES (SB, sample hands)")
    print("=" * 60)

    hand_samples = [
        "AhAs", "KhKs", "AhKh", "AhKd", "QhQs",
        "JhTs", "ThJs", "9h8h", "7h6h", "2h3d",
    ]

    for hand_str in hand_samples:
        c1, c2 = hand_str[:2], hand_str[2:4]
        card1, card2 = (
            _rank_suit_to_card(c1),
            _rank_suit_to_card(c2),
        )
        # Build a deal with these cards
        used = {card1, card2}
        board = []
        for c in range(52):
            if c not in used and len(board) < 5:
                board.append(c)
                used.add(c)
            if c not in used and len(board) >= 5:
                opp = [c]
                used.add(c)
                break
        # Find second opp card
        for c in range(52):
            if c not in used:
                opp.append(c)
                break

        h = ((card1, card2), tuple(opp), tuple(board))
        strat = solver.get_strategy(h, 0)
        actions = game.legal_actions(h)
        parts = [f"{a}={strat[i]:.0%}" for i, a in enumerate(actions)]
        name = f"{card_to_str(card1)}{card_to_str(card2)}"
        print(f"  {name:>6s}: {'  '.join(parts)}")

    # Save model
    torch.save(solver.strategy_net.state_dict(), "deep_cfr_strategy.pt")
    torch.save(solver.regret_net.state_dict(), "deep_cfr_regret.pt")
    print(f"\nModels saved: deep_cfr_strategy.pt, deep_cfr_regret.pt")


def _rank_suit_to_card(s: str) -> int:
    """Convert 'Ah' → card int."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks.index(s[0]) * 4 + suits.index(s[1])


if __name__ == "__main__":
    main()