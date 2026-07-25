#!/usr/bin/env python3
"""
Phase B.2 measurement: how many strength buckets a flop-level value net
needs to represent per-holding CFVs without losing accuracy.

The river/turn nets use K=50. This probe quantifies the trade-off: for a
fixed set of exactly-solved river spots, it takes the exact per-combo
counterfactual values and asks how much a K-bucket representation loses.
For each K it reconstructs a bucketed value vector (per combo = its
bucket's own-range-weighted mean) and measures the L1 distance to the
exact per-combo values, plus the marginal delta to the next-larger K.

The memo picks the smallest K whose delta-to-next is below the stated
threshold. Cheap: a handful of exact river solves, no training.

Usage:
    python3 scripts/flop_bucket_sensitivity.py --spots 8 --ks 20,30,50,80
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.games.postflop_nlhe import PostflopNLHE
from src.search.nlhe_river_vector import (
    RiverVectorCFR, COMBOS, N_COMBOS, bucket_map, bucket_values,
)


def _rand_range(rng, board, n=180):
    bs = set(board)
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    idx = rng.choice(live, size=n, replace=False)
    v = np.zeros(N_COMBOS)
    v[idx] = rng.dirichlet(np.ones(n))
    return v


def bucketed_reconstruction(v_combo, x_own, board, k):
    """Per-combo values → K-bucket own-range-weighted means → broadcast
    back to per-combo. This is exactly the lossy representation the value
    net is trained to output."""
    bm = bucket_map(board, k)
    bv = bucket_values(v_combo, x_own, bm, k)   # [k]
    recon = np.zeros(N_COMBOS)
    ok = bm >= 0
    recon[ok] = bv[bm[ok]]
    return recon, bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=8)
    ap.add_argument("--ks", default="20,30,50,80")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    rng = np.random.default_rng(args.seed)
    game = PostflopNLHE(starting_stack=50.0, max_raises_per_street=1,
                        raise_fractions=(0.5, 1.0))

    print(f"# Flop bucket-sensitivity probe  ({time.strftime('%Y-%m-%d %H:%M')})")
    print(f"# {args.spots} exact river solves, K ∈ {ks}, {args.iters} iters/solve")
    print("# metric: mass-weighted L1(bucketed − exact) per-combo value, pot units\n")

    # Random distinct 5-card boards.
    per_k = {k: [] for k in ks}
    t0 = time.time()
    for s in range(args.spots):
        deck = rng.permutation(52)
        board = tuple(int(c) for c in deck[:5])
        node = ((int(deck[5]), int(deck[6])), (int(deck[7]), int(deck[8])),
                board, "k", "c", "c", "c")
        x0, x1 = _rand_range(rng, board), _rand_range(rng, board)
        solver = RiverVectorCFR(game, node, x0, x1, iterations=args.iters)
        solver.solve()
        v0, _ = solver.root_values()
        # mass-weighted L1 over hero's in-support combos
        w = x0 / x0.sum()
        for k in ks:
            recon, bm = bucketed_reconstruction(v0, x0, board, k)
            l1 = float(np.sum(w * np.abs(recon - v0)))
            per_k[k].append(l1)
        if (s + 1) % 4 == 0:
            print(f"  [{s+1}/{args.spots}] {time.time()-t0:.0f}s")

    print(f"\n{'K':>4} {'mean L1':>10} {'Δ to next K':>14}")
    means = {k: float(np.mean(per_k[k])) for k in ks}
    for i, k in enumerate(ks):
        delta = "" if i == 0 else f"{means[ks[i-1]] - means[k]:+.4f}"
        print(f"{k:>4} {means[k]:>10.4f} {delta:>14}")
    print("\n# smaller K = coarser net = cheaper; pick smallest K whose")
    print("# improvement from the next-larger K is below the memo threshold.")


if __name__ == "__main__":
    raise SystemExit(main())
