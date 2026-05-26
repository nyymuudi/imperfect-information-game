#!/usr/bin/env python3
"""
Train Deep CFR on Postflop Heads-Up NLHE.

Usage:
    python3 -m src.deep_cfr.train_postflop [--iterations N] [--traversals T]
"""

import argparse
import time
import torch
import numpy as np
from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.deep_cfr_solver import DeepCFRSolver
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.cpp_backend import export_for_libtorch


def main():
    parser = argparse.ArgumentParser(description="Deep CFR on Postflop NLHE")
    parser.add_argument("--iterations",      "-n", type=int,   default=50)
    parser.add_argument("--traversals",      "-t", type=int,   default=200)
    parser.add_argument("--hidden",               type=int,   default=128)
    parser.add_argument("--buffer",               type=int,   default=500000)
    parser.add_argument("--strategy-buffer",      type=int,   default=0,
                        help="Strategy buffer capacity (0 = same as --buffer)")
    parser.add_argument("--stack",                type=float, default=200.0)
    parser.add_argument("--raise-fraction",       type=float, default=0.75,
                        help="Raise size as fraction of pot (default: 0.75)")
    parser.add_argument("--epochs",               type=int,   default=20)
    args = parser.parse_args()

    game    = PostflopNLHE(starting_stack=args.stack, max_raises_per_street=2,
                           raise_fractions=(args.raise_fraction,))
    encoder = NLHEEncoder(starting_stack=args.stack)

    solver = DeepCFRSolver(
        game=game,
        encoder=encoder,
        max_actions=4,
        buffer_capacity=args.buffer,
        strategy_buffer_capacity=args.strategy_buffer,
        hidden_size=args.hidden,
        train_epochs=args.epochs,
        train_batch=256,
        traversals_per_iter=args.traversals,
        use_cpp_engine=True,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        lr=1e-3,
    )

    strat_cap = args.strategy_buffer or args.buffer
    print(f"Deep CFR on HU Postflop NLHE")
    print(f"  Stack: {args.stack:.0f}BB | Raise: {args.raise_fraction:.0%} pot | Iterations: {args.iterations}")
    print(f"  Traversals/iter: {args.traversals} | Hidden: {args.hidden}")
    print(f"  Buffer: regret={args.buffer:,}  strategy={strat_cap:,} | Train epochs: {args.epochs}\n")

    t0 = time.time()

    def cb(s, i):
        elapsed = time.time() - t0
        rate = i * args.traversals * 2 / elapsed
        print(f"  iter {i:>4d}: MR={len(s.regret_buffer):>7,}, "
              f"MΠ={len(s.strategy_buffer):>7,}, "
              f"time={elapsed:>6.1f}s, {rate:.0f} trav/s")

    solver.solve(
        iterations=args.iterations,
        callback=cb,
        callback_freq=max(1, args.iterations // 20),
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Samples: MR={len(solver.regret_buffer):,}, MΠ={len(solver.strategy_buffer):,}\n")

    # Export strategy network
    strategy_path = "/tmp/cfr_strategy_net.pt"
    export_for_libtorch(solver.strategy_net).save(strategy_path)
    solver._cpp._engine.load_strategy_model(strategy_path)
    print("Strategy model loaded into C++ engine.")

    ACTION_NAMES = ["f", "k", "r", "a"]
    print("=" * 60)
    print("PREFLOP OPENING STRATEGIES (SB, sample hands)")
    print("=" * 60)

    hand_samples = [
        "AhAs", "KhKs", "AhKh", "AhKd", "QhQs",
        "JhTs", "9h8h", "Kd4s", "9s3d", "7h2d",
    ]
    for hand_str in hand_samples:
        card1 = _rank_suit_to_card(hand_str[:2])
        card2 = _rank_suit_to_card(hand_str[2:4])
        probs = solver._cpp._engine.query_preflop_strategy(card1, card2)
        parts = [f"{a}={probs[i]:.0%}" for i, a in enumerate(ACTION_NAMES)]
        print(f"  {hand_str:>6s}: {'  '.join(parts)}")

    torch.save(solver.strategy_net.state_dict(), "deep_cfr_strategy.pt")
    torch.save(solver.regret_net.state_dict(),   "deep_cfr_regret.pt")
    print(f"\nModels saved: deep_cfr_strategy.pt, deep_cfr_regret.pt")


def _rank_suit_to_card(s: str) -> int:
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks.index(s[0]) * 4 + suits.index(s[1])


if __name__ == "__main__":
    main()