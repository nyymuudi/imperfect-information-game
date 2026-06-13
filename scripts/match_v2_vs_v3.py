#!/usr/bin/env python3
"""
Head-to-head: blueprints/50bb_validation_v2_subcat vs 50bb_validation_v3_coarse.

Usage:
    python3 scripts/match_v2_vs_v3.py [--n-hands 5000]

Positive win_rate_mbb → v2_subcat beats v3_coarse.
Negative → v3_coarse beats v2_subcat.

NOTE: encoder follows the CURRENT codebase (FEATURE_QUANT=0.05). v2_subcat was
trained on FEATURE_QUANT=1e-6, so its querying state will be slightly off-grid
from what it learned on — this is the realistic deployment scenario after the
abstraction change, and what we actually want to measure.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.games.postflop_nlhe import PostflopNLHE
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.blueprint import Blueprint
from src.analysis.head_to_head import match, match_crn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-hands", type=int, default=5000)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--a",       default="blueprints/50bb_validation_v2_subcat",
                   help="Blueprint A path (default: v2_subcat)")
    p.add_argument("--b",       default="blueprints/50bb_validation_v3_coarse",
                   help="Blueprint B path (default: v3_coarse)")
    p.add_argument("--a-scheme", choices=["flat", "tree", "super", "tree42"], default="flat",
                   help="Board-bucket scheme for A's encoder (default: flat).")
    p.add_argument("--b-scheme", choices=["flat", "tree", "super", "tree42"], default="flat",
                   help="Board-bucket scheme for B's encoder (default: flat).")
    p.add_argument("--crn", action="store_true",
                   help="Use Common Random Numbers paired comparison.")
    p.add_argument("--ev-adjusted", action="store_true",
                   help="Replace post-all-in showdown outcomes with equity over "
                        "remaining board cards. Eliminates runout variance for "
                        "all-in spots — typical 2-5× reduction.")
    args = p.parse_args()

    bp_a = Blueprint.load(args.a, device="cpu")
    bp_b = Blueprint.load(args.b, device="cpu")

    print(f"A: {args.a}")
    print(f"   stack={bp_a.metadata.starting_stack:.0f}BB iters={bp_a.metadata.iterations}")
    print(f"B: {args.b}")
    print(f"   stack={bp_b.metadata.starting_stack:.0f}BB iters={bp_b.metadata.iterations}")

    game = PostflopNLHE(
        starting_stack=bp_a.metadata.starting_stack,
        max_raises_per_street=bp_a.metadata.max_raises,
        raise_fractions=(bp_a.metadata.raise_fraction,),
    )

    # Derive K_BOARD from each blueprint's state_size so K=8 and K=16
    # blueprints can play each other through their native encoders.
    # state_size = 28 + K_BOARD.
    def k_board_for(state_size: int) -> int:
        return state_size - 28

    k_a = k_board_for(bp_a.metadata.state_size)
    k_b = k_board_for(bp_b.metadata.state_size)
    print(f"   A K_BOARD={k_a}, scheme={args.a_scheme}, state_size={bp_a.metadata.state_size}")
    print(f"   B K_BOARD={k_b}, scheme={args.b_scheme}, state_size={bp_b.metadata.state_size}")
    enc_a = NLHEEncoder(starting_stack=bp_a.metadata.starting_stack,
                        K_BOARD=k_a, bucket_scheme=args.a_scheme)
    enc_b = NLHEEncoder(starting_stack=bp_b.metadata.starting_stack,
                        K_BOARD=k_b, bucket_scheme=args.b_scheme)

    if args.crn:
        # CRN treats --n-hands as pair count (each pair = 2 hands).
        n_pairs = args.n_hands
        print(f"\nCRN paired comparison, {n_pairs:,} pairs = "
              f"{n_pairs*2:,} hands (seed={args.seed})...\n")
        result = match_crn(bp_a, bp_b, game, enc_a, enc_b,
                           n_pairs=n_pairs, seed=args.seed, progress=True,
                           ev_adjusted=args.ev_adjusted)
    else:
        print(f"\nPlaying {args.n_hands:,} hands (seed={args.seed})...\n")
        result = match(bp_a, bp_b, game, enc_a, enc_b,
                       n_hands=args.n_hands, seed=args.seed, progress=True,
                       ev_adjusted=args.ev_adjusted)

    wr  = result["win_rate_mbb"]
    se  = result["stderr_mbb"]
    z   = wr / se if se > 0 else float("inf")
    label = "paired Δ" if args.crn else "win rate"
    n_str = (f"{result.get('n_pairs', 0):,} pairs"
             if args.crn else f"{result['n_hands']:,} hands")
    print()
    print(f"  A {label} vs B: {wr:+8.2f}  ± {se:5.2f} mbb/hand (n={n_str})")
    print(f"  z-score:         {z:+8.2f}")
    if abs(z) > 2.0:
        winner = "A" if wr > 0 else "B"
        print(f"  → statistically significant: {winner} wins")
    else:
        print(f"  → not significant at 95% (|z| < 2). More hands needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
