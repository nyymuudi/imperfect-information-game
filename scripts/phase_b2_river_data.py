#!/usr/bin/env python3
"""
Stage B2: generate river-boundary CFV training data (ReBeL data step).

Samples river-root PBSs two ways and solves each exactly with the
combo-level river vector CFR (validated vs the deal solver: mean
|Δv| = 0.007 chips, ~700× faster):

  * self-play: v16c blueprint plays to the first river decision; both
    ranges are Bayes-updated blueprint ranges (src/search/nlhe_pbs).
  * random:    fresh deal walked by RANDOM public actions to the river,
    ranges drawn Dirichlet(α ∈ {0.15, 0.3, 1.0, 3.0}) over live combos —
    covers the sharp solver-iterate ranges the turn DL solver will
    actually query (measured necessity on Leduc: on-policy MAE 2.2×).

Targets: pot-normalised 2×50-bucket CFVs. Saved as npz shards
(validation_runs/b2_river_data/shard_XXX.npz) so the run is resumable
and thermal-throttle-safe.

Usage:
    python3 scripts/phase_b2_river_data.py --n-samples 5000
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.search.nlhe_pbs import compute_ranges
from src.search.nlhe_river_vector import COMBOS, N_COMBOS, combo_range_to_vector
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS, river_target
from scripts.resolve_h2h import BlueprintPolicy, load_bundle
from src.analysis.exploitability import _sample_deal

RIVER_STREET = 3


def walk_to_river(game, act_fn, deal, rng, max_steps=40):
    """Follow act_fn to the FIRST river decision; None if hand ended."""
    h = deal
    for _ in range(max_steps):
        if game.is_terminal(h):
            return None
        if game._parse_state(h)["street_idx"] == RIVER_STREET:
            return h
        player = game.current_player(h)
        actions = game.legal_actions(h)
        probs = act_fn(h, player, actions)
        h = game.apply_action(
            h, actions[int(rng.choice(len(actions), p=probs))])
    return None


def random_combo_range(rng, board):
    alpha = float(rng.choice([0.15, 0.3, 1.0, 3.0]))
    bs = set(board)
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    support = rng.choice(live, size=int(rng.integers(30, 400)), replace=False)
    v = np.zeros(N_COMBOS)
    v[support] = rng.dirichlet(np.full(len(support), alpha))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="blueprints/50bb_v16c_2size_cache_2500")
    ap.add_argument("--cfr-cache", default="blueprints/cache/v16b_advisor_v1.cache.npz")
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--selfplay-frac", type=float, default=0.3)
    ap.add_argument("--solve-iters", type=int, default=250)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="validation_runs/b2_river_data")
    args = ap.parse_args()

    bp, game, encoder = load_bundle(args.blueprint, args.cfr_cache)
    bpol = BlueprintPolicy(bp, encoder)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    existing = sorted(Path(args.out_dir).glob("shard_*.npz"))
    n_done = sum(int(np.load(p)["X"].shape[0]) for p in existing)
    shard_no = len(existing)
    print(f"[resume] {n_done} samples in {shard_no} shards")

    def bp_act(h, player, actions):
        return bpol.act(h, player, game)

    def rand_act(h, player, actions):
        return np.ones(len(actions)) / len(actions)

    X, Y = [], []
    t0 = time.time()
    made = n_done
    while made < args.n_samples:
        use_selfplay = rng.random() < args.selfplay_frac
        deal = _sample_deal(rng)
        node = walk_to_river(game, bp_act if use_selfplay else rand_act,
                             deal, rng)
        if node is None:
            continue
        board = tuple(node[2])
        try:
            if use_selfplay:
                r0, r1 = compute_ranges(bp, encoder, game, node)
                x0 = combo_range_to_vector(r0, board)
                x1 = combo_range_to_vector(r1, board)
            else:
                x0 = random_combo_range(rng, board)
                x1 = random_combo_range(rng, board)
            enc, tgt = river_target(game, node, x0, x1,
                                    solve_iters=args.solve_iters)
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")
            continue
        X.append(enc)
        Y.append(tgt)
        made += 1

        if len(X) >= args.shard_size:
            path = os.path.join(args.out_dir, f"shard_{shard_no:03d}.npz")
            np.savez_compressed(path, X=np.asarray(X), Y=np.asarray(Y))
            rate = (made - n_done) / (time.time() - t0)
            eta = (args.n_samples - made) / max(rate, 1e-9) / 3600
            print(f"  [{made}/{args.n_samples}] wrote {path} "
                  f"({rate:.2f} samples/s, ETA {eta:.1f}h)", flush=True)
            X, Y = [], []
            shard_no += 1

    if X:
        path = os.path.join(args.out_dir, f"shard_{shard_no:03d}.npz")
        np.savez_compressed(path, X=np.asarray(X), Y=np.asarray(Y))
        print(f"  final shard: {path}")
    print(f"done: {made} samples, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
