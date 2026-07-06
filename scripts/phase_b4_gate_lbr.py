#!/usr/bin/env python3
"""
Stage B gate: whole-strategy LBR of the depth-limited search agent.

The agent plays v16c everywhere except turn and river decisions, which
are re-solved at decision time with Bayes ranges (nlhe_pbs):
    river  → exact combo-level RiverVectorCFR
    turn   → depth-limited TurnVectorCFR with the B4 river-CFV net

Measurement: estimate_exploitability with strategy_override, SAME seed
and config as the plain-blueprint run → the trajectories, evaluated
nodes and opponent samples are identical (the override draws nothing
from the shared rng), so Δ is CRN-paired.

This is the gate the Phase 3 river-null diagnostics prescribed: the
best-responder faces the ACTUAL mixed agent at the evaluated node,
instead of the per-node one-shot metric that cannot reward balanced
re-solving (see project memory / phase3 diagnostics).

Usage:
    python3 scripts/phase_b4_gate_lbr.py --n-games 400
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.analysis.exploitability import estimate_exploitability
from src.search.cfv_net import CFVNet
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS
from src.search.nlhe_pbs import compute_ranges
from src.search.nlhe_river_vector import RiverVectorCFR, combo_range_to_vector
from src.search.nlhe_turn_vector import TurnVectorCFR
from scripts.resolve_h2h import load_bundle

TURN, RIVER = 2, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="blueprints/50bb_v16c_2size_cache_2500")
    ap.add_argument("--cfr-cache", default="blueprints/cache/v16b_advisor_v1.cache.npz")
    ap.add_argument("--net", default="validation_runs/b4_river_cfv_net.pt")
    ap.add_argument("--n-games", type=int, default=400)
    ap.add_argument("--river-iters", type=int, default=300)
    ap.add_argument("--turn-iters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    bp, game, encoder = load_bundle(args.blueprint, args.cfr_cache)
    net = CFVNet(input_dim=RIVER_ENCODING_DIMS, output_dim=100, hidden=256)
    net.load_state_dict(torch.load(args.net))
    net.eval()

    stats = {"turn": 0, "river": 0, "skip": 0, "fail": 0}

    def override(history, hero, g):
        st = g._parse_state(history)
        street = st["street_idx"]
        if street not in (TURN, RIVER) or st["all_in"]:
            stats["skip"] += 1
            return None
        try:
            r0, r1 = compute_ranges(bp, encoder, g, history)
            n_vis = (0, 3, 4, 5)[min(street, 3)]
            board = tuple(history[2][:n_vis])
            x0 = combo_range_to_vector(r0, board)
            x1 = combo_range_to_vector(r1, board)
            if street == RIVER:
                solver = RiverVectorCFR(g, history, x0, x1,
                                        iterations=args.river_iters)
                solver.solve()
                stats["river"] += 1
            else:
                solver = TurnVectorCFR(g, history, x0, x1, net,
                                       iterations=args.turn_iters)
                solver.solve()
                stats["turn"] += 1
            return solver.strategy_at(history, hero)
        except Exception as e:
            stats["fail"] += 1
            print(f"  [override-fail] {type(e).__name__}: {e}", flush=True)
            return None

    t0 = time.time()
    print(f"[1/2] baseline: v16c plain (n={args.n_games}, seed={args.seed})")
    base = estimate_exploitability(bp, game, encoder, n_games=args.n_games,
                                   seed=args.seed, verbose=False)
    print(f"      LBR = {float(base):.1f} ± {base.stderr_mbb:.1f} mbb/dec "
          f"(n={base.n_decisions}, {time.time()-t0:.0f}s)")
    if args.baseline_only:
        return 0

    t0 = time.time()
    print(f"[2/2] search agent: turn+river re-solve override")
    agent = estimate_exploitability(bp, game, encoder, n_games=args.n_games,
                                    seed=args.seed, verbose=False,
                                    strategy_override=override)
    print(f"      LBR = {float(agent):.1f} ± {agent.stderr_mbb:.1f} mbb/dec "
          f"(n={agent.n_decisions}, {time.time()-t0:.0f}s)")
    print(f"      resolves: turn={stats['turn']} river={stats['river']} "
          f"skip={stats['skip']} fail={stats['fail']}")

    d = float(agent) - float(base)
    se = (base.stderr_mbb ** 2 + agent.stderr_mbb ** 2) ** 0.5
    print(f"\n== Stage B gate ==")
    print(f"Δ(search − blueprint) = {d:+.1f} ± {se:.1f} mbb/dec "
          f"(CRN-paired trajectories)")
    print("PASS" if d < 0 else "no improvement")
    return 0 if d < 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
