#!/usr/bin/env python3
"""
Turn CFV-net training data (flop-level groundwork, per docs/flop_design.md).

Flop depth-limited solving truncates at the TURN boundary and needs
turn-root CFVs — a TURN value net that does not exist yet. This collector
generates its base training data: sample turn decision PBSs, solve each
with TurnVectorCFR (b5 river-net leaves, 23×-accelerated), and store the
turn-root per-combo values from TurnVectorCFR.root_values().

Values are stored RAW (per-combo, 1326-dim), not bucketed: the turn-root
bucketing scheme (a 6-card strength metric — evaluate_7card is 7-card
only) is an open design decision (see memo). Storing raw lets that
decision be made at train time without re-running the solves.

Resumable npz shards. Sampling: blueprint self-play Bayes ranges (via
nlhe_pbs) + random Dirichlet ranges for solver-iterate coverage —
the same recipe that fixed the on-policy gap on the river net.

Usage:
    python3 scripts/phase_c1_turn_data.py --n-samples 25000
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.search.cfv_net import CFVNet
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS
from src.search.nlhe_pbs import compute_ranges
from src.search.nlhe_river_vector import COMBOS, N_COMBOS, combo_range_to_vector
from src.search.nlhe_turn_vector import TurnVectorCFR, TURN_STREET
from src.analysis.exploitability import _sample_deal
from scripts.resolve_h2h import BlueprintPolicy, load_bundle

FLOP_STREET = 1


def walk_to_turn(game, bpol, deal, rng, max_steps=40):
    h = deal
    for _ in range(max_steps):
        if game.is_terminal(h):
            return None
        st = game._parse_state(h)
        if st["street_idx"] == TURN_STREET and not st["all_in"]:
            return h
        if st["street_idx"] > TURN_STREET:
            return None
        player = game.current_player(h)
        actions = game.legal_actions(h)
        probs = bpol.act(h, player, game)
        h = game.apply_action(h, actions[int(rng.choice(len(actions), p=probs))])
    return None


def random_combo_range(rng, board4):
    alpha = float(rng.choice([0.15, 0.3, 1.0, 3.0]))
    bs = set(board4)
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    n = int(rng.integers(30, 400))
    support = rng.choice(live, size=n, replace=False)
    v = np.zeros(N_COMBOS)
    v[support] = rng.dirichlet(np.full(n, alpha))
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="blueprints/50bb_v16c_2size_cache_2500")
    ap.add_argument("--cfr-cache", default="blueprints/cache/v16b_advisor_v1.cache.npz")
    ap.add_argument("--net", default="validation_runs/b5_river_cfv_net.pt")
    ap.add_argument("--n-samples", type=int, default=25000)
    ap.add_argument("--selfplay-frac", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="validation_runs/turn_cfv_data")
    args = ap.parse_args()

    bp, game, encoder = load_bundle(args.blueprint, args.cfr_cache)
    bpol = BlueprintPolicy(bp, encoder)
    net = CFVNet(input_dim=RIVER_ENCODING_DIMS, output_dim=100, hidden=256)
    net.load_state_dict(torch.load(args.net))
    net.eval()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    existing = sorted(Path(args.out_dir).glob("turn_*.npz"))
    n_done = sum(int(np.load(p)["board4"].shape[0]) for p in existing)
    shard_no = len(existing)
    print(f"[resume] {n_done} samples in {shard_no} shards", flush=True)

    boards, pots, r0s, r1s, v0s, v1s = [], [], [], [], [], []
    made = n_done
    t0 = time.time()
    while made < args.n_samples:
        use_sp = rng.random() < args.selfplay_frac
        deal = _sample_deal(rng)
        node = walk_to_turn(game, bpol, deal, rng)
        if node is None:
            continue
        board4 = tuple(int(c) for c in node[2][:4])
        try:
            if use_sp:
                rr0, rr1 = compute_ranges(bp, encoder, game, node)
                x0 = combo_range_to_vector(rr0, board4)
                x1 = combo_range_to_vector(rr1, board4)
            else:
                x0 = random_combo_range(rng, board4)
                x1 = random_combo_range(rng, board4)
            solver = TurnVectorCFR(game, node, x0, x1, net, iterations=args.iters)
            solver.solve()
            v0, v1 = solver.root_values()
            pot = float(game._parse_state(node)["pot"])
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}", flush=True)
            continue

        boards.append(board4)
        pots.append(pot)
        r0s.append(x0.astype(np.float32)); r1s.append(x1.astype(np.float32))
        v0s.append(v0.astype(np.float32)); v1s.append(v1.astype(np.float32))
        made += 1

        if len(boards) >= args.shard_size:
            _flush(args.out_dir, shard_no, boards, pots, r0s, r1s, v0s, v1s)
            shard_no += 1
            rate = (made - n_done) / (time.time() - t0) * 3600
            eta = (args.n_samples - made) / max(rate, 1e-9)
            print(f"  [{made}/{args.n_samples}] {rate:.0f}/h  ETA {eta:.1f}h",
                  flush=True)
            boards, pots, r0s, r1s, v0s, v1s = [], [], [], [], [], []

    if boards:
        _flush(args.out_dir, shard_no, boards, pots, r0s, r1s, v0s, v1s)
    print(f"done: {made} samples, {time.time()-t0:.0f}s", flush=True)
    return 0


def _flush(out_dir, shard_no, boards, pots, r0s, r1s, v0s, v1s):
    path = os.path.join(out_dir, f"turn_{shard_no:04d}.npz")
    np.savez_compressed(
        path,
        board4=np.asarray(boards, dtype=np.int16),
        pot=np.asarray(pots, dtype=np.float32),
        r0=np.asarray(r0s), r1=np.asarray(r1s),
        v0=np.asarray(v0s), v1=np.asarray(v1s),
    )
    print(f"  wrote {path} ({len(boards)} samples)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
