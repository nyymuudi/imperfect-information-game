#!/usr/bin/env python3
"""
Phase 3 gate measurement: PBS-range river re-solving vs blueprint (NLHE).

Paired per-river-node LBR regret — the dense, CRN methodology from
resolve_river_lbr.py (whole-strategy LBR dilutes river effects below the
noise floor; see that script's docstring). New vs the June experiments:

  * BOTH ranges are Bayes-updated blueprint ranges (src/search/nlhe_pbs),
    not hero-pinned + opponent-uniform. The June recipe measured Δ≈0 —
    uniform ranges carry no information about the observed line.
  * The shared action EVs marginalise the opponent by their Bayes range
    (reach-weighted), not uniformly. Validated on Leduc: the uniform
    belief floors near equilibrium (phase3_lbr_leduc_validation.py).

For each sampled river decision node (reached by blueprint self-play):
    action_evs  — _lbr_action_evs with the Bayes opponent sampler (shared
                  across arms; paired deltas cancel the EV noise)
    regret(π)   = max_a EV[a] − π·EV        for each arm:
        bp      — blueprint policy at the node
        pbs     — UnsafeSubgameSolver with Bayes ranges (both players)
        unif    — June recipe (hero pinned, opponent uniform), reference

Exit signal: mean paired delta regret(pbs) − regret(bp) < 0 with
significance, and pbs < unif shows the ranges (not the solving) carry
the improvement.

Usage:
    python3 scripts/phase3_river_pbs_lbr.py --n-nodes 120
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.exploitability import _lbr_action_evs
from src.analysis.head_to_head import ev_adjusted_payoffs
from src.search.nlhe_pbs import compute_ranges, range_sampler, top_k_range
from src.solvers.subgame_solver import UnsafeSubgameSolver
from scripts.resolve_h2h import (
    BlueprintPolicy, load_bundle, _uniform_range, _resolve_seed,
)
from scripts.resolve_river_lbr import _walk_to_river_decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default="blueprints/50bb_v15_c2_v5_aux")
    ap.add_argument("--cfr-cache",
                    default="blueprints/cache/v14d_advisor_v4.cache.npz")
    ap.add_argument("--n-nodes", type=int, default=120)
    ap.add_argument("--iterations", type=int, default=150,
                    help="subgame CFR iterations per resolve")
    ap.add_argument("--max-deals", type=int, default=150)
    ap.add_argument("--range-k", type=int, default=80,
                    help="top-K combos kept per range for the resolve")
    ap.add_argument("--n-opp", type=int, default=16,
                    help="opponent samples for the shared action EVs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bp, game, encoder = load_bundle(args.blueprint, args.cfr_cache)
    bpol = BlueprintPolicy(bp, encoder)
    solver = UnsafeSubgameSolver(game)
    rng = np.random.default_rng(args.seed)

    from src.analysis.exploitability import _sample_deal

    reg = {"bp": [], "pbs": [], "unif": []}
    t0 = time.time()
    found, attempts = 0, 0
    max_attempts = args.n_nodes * 400

    while found < args.n_nodes and attempts < max_attempts:
        attempts += 1
        deal = _sample_deal(rng)
        node = _walk_to_river_decision(bpol, game, deal, rng)
        if node is None:
            continue

        hero = game.current_player(node)
        opp = 1 - hero
        hero_cards = tuple(int(c) for c in node[hero])
        board = node[2]

        # Bayes ranges at this node (both players).
        try:
            ranges = compute_ranges(bp, encoder, game, node)
        except Exception as e:
            print(f"  [skip] compute_ranges failed: {e}")
            continue
        hero_range = top_k_range(ranges[hero], args.range_k,
                                 must_include=hero_cards, floor=0.01)
        opp_range = top_k_range(ranges[opp], args.range_k)

        # Shared action EVs: Bayes-weighted opponent marginalisation.
        sampler = range_sampler(
            ranges[opp], exclude=set(hero_cards) | set(board))
        if sampler is None:
            continue
        action_evs = _lbr_action_evs(
            bp, encoder, game, node, hero, opp, rng,
            n_opp_samples=args.n_opp, payoff_fn=ev_adjusted_payoffs,
            opp_sampler=sampler,
        )

        actions = game.legal_actions(node)
        n = len(actions)
        best = float(action_evs.max())

        def regret_of(probs) -> float:
            p = np.asarray(probs, dtype=np.float64)[:n]
            s = p.sum()
            p = p / s if s > 1e-12 else np.ones(n) / n
            return max(0.0, best - float(p @ action_evs))

        # Arm 1: blueprint.
        r_bp = regret_of(bpol.act(node, hero, game))

        # Arm 2: PBS resolve (Bayes ranges both sides).
        seed = _resolve_seed(node)
        strat = solver.solve(
            root_history=node, hero_player=hero,
            hero_range=hero_range, opponent_range=opp_range,
            iterations=args.iterations, max_deals=args.max_deals,
            rng=np.random.default_rng(seed),
        )
        r_pbs = regret_of(strat.query(node, hero))

        # Arm 3: June recipe (hero pinned, opponent uniform).
        u_range = _uniform_range(board, exclude=node[hero])
        strat_u = solver.solve(
            root_history=node, hero_player=hero,
            hero_range={tuple(node[hero]): 1.0}, opponent_range=u_range,
            iterations=args.iterations, max_deals=args.max_deals,
            rng=np.random.default_rng(seed),
        )
        r_unif = regret_of(strat_u.query(node, hero))

        reg["bp"].append(r_bp)
        reg["pbs"].append(r_pbs)
        reg["unif"].append(r_unif)
        found += 1
        if found % 20 == 0:
            d = np.asarray(reg["pbs"]) - np.asarray(reg["bp"])
            print(f"  [{found}/{args.n_nodes}] paired Δ(pbs−bp) = "
                  f"{d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
                  f"chips  ({time.time()-t0:.0f}s)")

    print(f"\nnodes={found} (attempts={attempts}, {time.time()-t0:.0f}s)")
    bb = 2.0
    for arm in ("bp", "unif", "pbs"):
        a = np.asarray(reg[arm])
        print(f"  regret[{arm:<4}] = {a.mean():.4f} chips/node "
              f"({1000*a.mean()/bb:.1f} mbb)")

    print("\npaired deltas (negative = arm less exploitable than blueprint):")
    for arm in ("unif", "pbs"):
        d = np.asarray(reg[arm]) - np.asarray(reg["bp"])
        se = d.std(ddof=1) / np.sqrt(len(d))
        z = d.mean() / se if se > 0 else float("nan")
        print(f"  Δ({arm}−bp)  = {d.mean():+.4f} ± {se:.4f} chips "
              f"({1000*d.mean()/bb:+.1f} ± {1000*se/bb:.1f} mbb)  z={z:+.2f}")
    d = np.asarray(reg["pbs"]) - np.asarray(reg["unif"])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  Δ(pbs−unif) = {d.mean():+.4f} ± {se:.4f} chips  "
          f"(range information effect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
