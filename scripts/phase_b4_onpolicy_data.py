#!/usr/bin/env python3
"""
Stage B4 data: on-policy river-boundary targets from turn DL solves.

The Leduc Phase 2 failure→fix showed the value net must be trained on
the PBSs the depth-limited solver ACTUALLY queries (solver-iterate
ranges are sharper than blueprint self-play; on-policy MAE was 2.2×
off-policy). This script:

  1. samples turn decision nodes from v16c blueprint self-play,
  2. computes Bayes ranges (nlhe_pbs) and runs a TurnVectorCFR solve,
  3. replays the solve's FINAL average strategies to every
     turn-completing line × river card — exactly the leaf PBSs the
     converged solver queries — and solves each exactly,
  4. appends (encoding, pot-normalised bucket-CFV) shards next to the
     B2 base data (onpolicy_*.npz).

Usage:
    python3 scripts/phase_b4_onpolicy_data.py --n-turn-nodes 100
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
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS, river_target
from src.search.nlhe_pbs import compute_ranges
from src.search.nlhe_river_vector import combo_range_to_vector
from src.search.nlhe_turn_vector import TurnVectorCFR
from src.analysis.exploitability import _sample_deal
from scripts.resolve_h2h import BlueprintPolicy, load_bundle

TURN_STREET = 2


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
        h = game.apply_action(
            h, actions[int(rng.choice(len(actions), p=probs))])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="blueprints/50bb_v16c_2size_cache_2500")
    ap.add_argument("--cfr-cache", default="blueprints/cache/v16b_advisor_v1.cache.npz")
    ap.add_argument("--net", default="validation_runs/b2_river_cfv_net.pt")
    ap.add_argument("--n-turn-nodes", type=int, default=100)
    ap.add_argument("--queries-per-node", type=int, default=30)
    ap.add_argument("--dl-iters", type=int, default=150)
    ap.add_argument("--solve-iters", type=int, default=250)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out-dir", default="validation_runs/b2_river_data")
    args = ap.parse_args()

    bp, game, encoder = load_bundle(args.blueprint, args.cfr_cache)
    bpol = BlueprintPolicy(bp, encoder)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    net = CFVNet(input_dim=RIVER_ENCODING_DIMS, output_dim=100, hidden=256)
    net.load_state_dict(torch.load(args.net))
    net.eval()

    existing = sorted(Path(args.out_dir).glob("onpolicy_*.npz"))
    shard_no = len(existing)
    print(f"[resume] {shard_no} on-policy shards exist")

    X, Y = [], []
    done = 0
    t0 = time.time()
    while done < args.n_turn_nodes:
        deal = _sample_deal(rng)
        node = walk_to_turn(game, bpol, deal, rng)
        if node is None:
            continue
        try:
            r0, r1 = compute_ranges(bp, encoder, game, node)
            board4 = tuple(node[2][:4])
            x0 = combo_range_to_vector(r0, board4)
            x1 = combo_range_to_vector(r1, board4)
            solver = TurnVectorCFR(game, node, x0, x1, net,
                                   iterations=args.dl_iters)
            solver.solve()
            pot_actions = _final_boundary_pbs(game, solver, node, x0, x1)
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")
            continue
        if not pot_actions:
            continue
        picks2 = rng.choice(len(pot_actions),
                            size=min(args.queries_per_node, len(pot_actions)),
                            replace=False)
        for k in picks2:
            river_node, m0, m1 = pot_actions[k]
            try:
                enc, tgt = river_target(game, river_node, m0, m1,
                                        solve_iters=args.solve_iters)
            except Exception:
                continue
            X.append(enc)
            Y.append(tgt)

        done += 1
        rate = done / (time.time() - t0)
        print(f"  [{done}/{args.n_turn_nodes}] samples={len(X)} "
              f"({rate*3600:.0f} nodes/h)", flush=True)

        if len(X) >= 500:
            _flush(args.out_dir, shard_no, X, Y)
            shard_no += 1
            X, Y = [], []

    if X:
        _flush(args.out_dir, shard_no, X, Y)
    print(f"done: {done} turn nodes, {time.time()-t0:.0f}s")
    return 0


def _final_boundary_pbs(game, solver, node, x0, x1):
    """Enumerate (river_node, x0_leaf, x1_leaf) for every turn-completing
    line under the solver's final AVERAGE strategies × every river card."""
    out = []

    def sigma_at(cont, na):
        ss = solver._strat_sum.get((cont,))
        if ss is None:
            return np.full((len(x0), na), 1.0 / na)
        tot = ss.sum(axis=1, keepdims=True)
        return np.where(tot > 0, ss / np.where(tot > 0, tot, 1.0), 1.0 / na)

    def walk(cont, r0, r1):
        rep = solver._rep(cont)
        st = game._parse_state(rep)
        if game.is_terminal(rep) or st["street_idx"] > TURN_STREET:
            if st["folded"][0] or st["folded"][1] or st["all_in"]:
                return
            # Turn complete → one leaf PBS per river card.
            for c in solver.river_cards:
                mask = solver._card_mask[c]
                m0 = np.where(mask, r0, 0.0)
                m1 = np.where(mask, r1, 0.0)
                if m0.sum() <= 1e-9 or m1.sum() <= 1e-9:
                    continue
                board5 = solver.turn_board + (c,)
                river_node = (rep[0], rep[1], board5) + tuple(rep[3:])
                out.append((river_node, m0 / m0.sum(), m1 / m1.sum()))
            return
        player = game.current_player(rep)
        legal = game.legal_actions(rep)
        sig = sigma_at(cont, len(legal))
        for ai, a in enumerate(legal):
            if player == 0:
                walk(cont + (a,), r0 * sig[:, ai], r1)
            else:
                walk(cont + (a,), r0, r1 * sig[:, ai])

    walk((), x0.copy(), x1.copy())
    return out


def _flush(out_dir, shard_no, X, Y):
    path = os.path.join(out_dir, f"onpolicy_{shard_no:03d}.npz")
    np.savez_compressed(path, X=np.asarray(X, dtype=np.float32),
                        Y=np.asarray(Y, dtype=np.float32))
    print(f"  wrote {path} ({len(X)} samples)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
