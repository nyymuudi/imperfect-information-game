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
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma list; overrides --seed, aggregates with a "
                         "per-seed paired t-test")
    ap.add_argument("--opp-beliefs", choices=["uniform", "bayes"],
                    default="uniform")
    ap.add_argument("--depth-sampling", choices=["uniform", "stratified"],
                    default="uniform")
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

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [args.seed])
    metric = dict(opp_beliefs=args.opp_beliefs,
                  depth_sampling=args.depth_sampling)

    per_seed_delta, per_seed_pse = [], []
    for sd in seeds:
        t0 = time.time()
        base = estimate_exploitability(bp, game, encoder,
                                       n_games=args.n_games, seed=sd,
                                       verbose=False, **metric)
        print(f"seed {sd} baseline: {float(base):7.1f} ± {base.stderr_mbb:5.1f} "
              f"mbb/dec (n={base.n_decisions}, {time.time()-t0:.0f}s)")
        if args.baseline_only:
            continue

        t0 = time.time()
        agent = estimate_exploitability(bp, game, encoder,
                                        n_games=args.n_games, seed=sd,
                                        verbose=False,
                                        strategy_override=override, **metric)
        d = float(agent) - float(base)
        # Proper CRN-paired SE from aligned per-node samples (identical
        # node sets — override consumes no shared rng draws). Falls back
        # to quadrature only if alignment broke (differing skips).
        if (agent.samples_mbb is not None and base.samples_mbb is not None
                and len(agent.samples_mbb) == len(base.samples_mbb)):
            dd = agent.samples_mbb - base.samples_mbb
            if metric["depth_sampling"] == "stratified" \
                    and agent.strata_weights is not None:
                pse, dmean = 0.0, 0.0
                for s in range(4):
                    m = dd[agent.streets == s]
                    if len(m) == 0:
                        continue
                    dmean += agent.strata_weights[s] * float(m.mean())
                    if len(m) >= 2:
                        pse += (agent.strata_weights[s] ** 2
                                * float(m.var(ddof=1)) / len(m))
                d, pse = dmean, float(np.sqrt(pse))
            else:
                pse = float(dd.std(ddof=1) / np.sqrt(len(dd)))
        else:
            pse = (base.stderr_mbb ** 2 + agent.stderr_mbb ** 2) ** 0.5
        per_seed_delta.append(d)
        per_seed_pse.append(pse)
        print(f"seed {sd} agent   : {float(agent):7.1f} ± {agent.stderr_mbb:5.1f} "
              f"| paired Δ = {d:+7.1f} ± {pse:5.1f} mbb/dec "
              f"(turn={stats['turn']} river={stats['river']} "
              f"fail={stats['fail']}, {time.time()-t0:.0f}s)")
        stats.update({"turn": 0, "river": 0, "skip": 0, "fail": 0})

    if args.baseline_only or not per_seed_delta:
        return 0
    d = np.asarray(per_seed_delta)
    print(f"\n== Stage B gate ({len(d)} seeds, "
          f"beliefs={args.opp_beliefs}, sampling={args.depth_sampling}) ==")
    if len(d) >= 2:
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = d.mean() / se if se > 0 else float("nan")
        print(f"Δ(search − blueprint) = {d.mean():+.1f} ± {se:.1f} mbb/dec  "
              f"t = {t:+.2f}  ({int((d < 0).sum())}/{len(d)} seeds negative)")
    else:
        print(f"Δ(search − blueprint) = {d[0]:+.1f} ± {per_seed_pse[0]:.1f} "
              f"mbb/dec (single seed, paired SE)")
    print("PASS" if d.mean() < 0 else "no improvement")
    return 0 if d.mean() < 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
