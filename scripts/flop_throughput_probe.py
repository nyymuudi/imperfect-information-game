#!/usr/bin/env python3
"""
Phase B.1 measurement: flop-level solve throughput on this hardware.

A flop depth-limited solver truncates at the TURN boundary. Each turn
leaf needs a turn-root CFV — which today only a nested TurnVectorCFR
(b5 river net) can produce, since no turn CFV net exists yet. This probe
measures the two quantities the flop design memo needs:

  1. Turn-solve cost (the flop leaf-evaluation building block, and the
     turn-CFV-net data-generation cost) at 3 iteration budgets, in a
     COLD window and a sustained WARM window (thermal throttling on a
     fanless M2). Reports solves/hour.

  2. Flop betting-tree size: number of turn-boundary leaves reachable
     from a flop root under the v16c action set. Multiplied by the
     turn-leaf cost this bounds a NESTED flop solve; divided into a
     single batched net query it bounds a flop solve GIVEN a turn net.

Output is plain text → committed to docs/experiments/ as the memo source.

Usage:
    python3 scripts/flop_throughput_probe.py --warm-minutes 60
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.games.postflop_nlhe import PostflopNLHE
from src.search.cfv_net import CFVNet
from src.search.nlhe_cfv import RIVER_ENCODING_DIMS
from src.search.nlhe_turn_vector import TurnVectorCFR, TURN_STREET
from src.search.nlhe_river_vector import COMBOS, N_COMBOS

FLOP_STREET = 1


def _rand_range(rng, board_prefix, n=200):
    bs = set(board_prefix)
    live = [i for i, c in enumerate(COMBOS) if not (bs & set(c))]
    idx = rng.choice(live, size=n, replace=False)
    v = np.zeros(N_COMBOS)
    v[idx] = rng.dirichlet(np.ones(n))
    return v


def time_turn_solve(game, net, node, iters, rng):
    board4 = tuple(node[2][:4])
    x0, x1 = _rand_range(rng, board4), _rand_range(rng, board4)
    t0 = time.time()
    solver = TurnVectorCFR(game, node, x0, x1, net, iterations=iters)
    t_setup = time.time() - t0
    t0 = time.time()
    solver.solve()
    return t_setup, time.time() - t0


def count_flop_turn_leaves(game):
    """Enumerate flop-betting continuations to the turn boundary from a
    flop decision node (v16c: 2 raise sizes → 6-slot action set).
    Counts distinct betting lines that COMPLETE the flop street (each is
    one turn-chance node = 46 turn cards to evaluate)."""
    # A concrete flop node: SB calls preflop, BB checks → flop, BB acts.
    board5 = (0, 5, 10, 15, 20)
    root = ((48, 49), (24, 1), board5, "k", "c")
    st = game._parse_state(root)
    assert st["street_idx"] == FLOP_STREET, st["street_idx"]

    leaves = {"lines": 0, "nodes": 0}

    def walk(h):
        leaves["nodes"] += 1
        st = game._parse_state(h)
        if game.is_terminal(h):
            return
        if st["street_idx"] > FLOP_STREET:
            # Flop betting completed → one turn-chance boundary.
            if not st["all_in"]:
                leaves["lines"] += 1
            return
        for a in game.legal_actions(h):
            walk(game.apply_action(h, a))

    walk(root)
    return leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="validation_runs/b5_river_cfv_net.pt")
    ap.add_argument("--iters", default="1000,2500,5000")
    ap.add_argument("--cold-samples", type=int, default=5)
    ap.add_argument("--warm-minutes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)
    game = PostflopNLHE(starting_stack=50.0, max_raises_per_street=1,
                        raise_fractions=(0.5, 1.0))
    net = CFVNet(input_dim=RIVER_ENCODING_DIMS, output_dim=100, hidden=256)
    net.load_state_dict(torch.load(args.net))
    net.eval()
    rng = np.random.default_rng(args.seed)

    # Turn decision node (b5 river-net leaves).
    board5 = (0, 5, 10, 15, 20)
    node = ((48, 49), (24, 1), board5, "k", "c", "c", "c")
    assert game._parse_state(node)["street_idx"] == TURN_STREET

    budgets = [int(x) for x in args.iters.split(",")]

    print(f"# Flop throughput probe  ({time.strftime('%Y-%m-%d %H:%M')})")
    print("# hardware: M2 Air (fanless), b5 river CFV net")
    print("# turn solve = flop leaf-eval building block\n")

    # ── Flop tree size ───────────────────────────────────────────────────────
    tree = count_flop_turn_leaves(game)
    print(f"[flop tree] betting nodes={tree['nodes']}  "
          f"turn-boundary lines={tree['lines']}  "
          f"(each line = 46 turn cards to evaluate)\n")

    # ── Cold window: per-budget turn-solve cost ──────────────────────────────
    print(f"[cold] {args.cold_samples} solves per budget (fresh process):")
    cold = {}
    for it in budgets:
        setups, solves = [], []
        for _ in range(args.cold_samples):
            ts, tv = time_turn_solve(game, net, node, it, rng)
            setups.append(ts)
            solves.append(tv)
        setup_m = float(np.median(setups))
        solve_m = float(np.median(solves))
        total = setup_m + solve_m
        cold[it] = total
        print(f"  iters={it:>5}: setup={setup_m:5.2f}s solve={solve_m:6.2f}s "
              f"total={total:6.2f}s  → {3600/total:6.0f} solves/h")
    print()

    # ── Warm window: sustained throughput (thermal) ──────────────────────────
    it = budgets[1] if len(budgets) > 1 else budgets[0]
    print(f"[warm] sustained solves at iters={it} for {args.warm_minutes} min "
          f"(thermal throttling):")
    deadline = time.time() + args.warm_minutes * 60
    n_done = 0
    win_start = time.time()
    win_first10 = None
    while time.time() < deadline:
        time_turn_solve(game, net, node, it, rng)
        n_done += 1
        el = time.time() - win_start
        if win_first10 is None and el >= 600:
            win_first10 = n_done
            print(f"  first 10 min: {n_done} solves "
                  f"→ {n_done/10*60:.0f} solves/h (cold-ish)")
    el = time.time() - win_start
    rate_h = n_done / el * 3600
    print(f"  full {el/60:.0f} min: {n_done} solves → {rate_h:.0f} solves/h "
          f"(sustained)")
    if win_first10:
        throttle = (win_first10 / 10 * 60) / rate_h
        print(f"  throttle factor (cold/sustained): {throttle:.2f}×")
    print()

    # ── Flop-solve extrapolation ─────────────────────────────────────────────
    print("[extrapolation]")
    warm_solve_s = 3600 / rate_h
    nested = tree["lines"] * 46 * warm_solve_s
    print("  NESTED flop solve (no turn net): "
          f"{tree['lines']}×46×{warm_solve_s:.1f}s = {nested/3600:.0f} h/solve "
          f"→ INFEASIBLE")
    print("  GIVEN a turn net, a flop solve ≈ one turn-class solve "
          f"(~{warm_solve_s:.1f}s) → same order as turn data-gen")
    print(f"  ⇒ memo implication: train a TURN CFV net first; its data = "
          f"turn solves at {rate_h:.0f}/h sustained")


if __name__ == "__main__":
    raise SystemExit(main())
