#!/usr/bin/env python3
"""scripts/distill_subgame.py

Two-in-one CLI for the subgame distiller:

  diagnose:  python3 scripts/distill_subgame.py diagnose <bp_path> [--n-spots N]
             → prints a GapReport (TV / argmax-disagreement / solved info-sets)

  distill:   python3 scripts/distill_subgame.py distill <bp_path> -o targets.npz [--n-spots N]
             → solves N random subgames, writes (state, strategy, n_legal)
                tuples to a single .npz the trainer can later boost-feed
                into the strategy buffer.

A typical workflow:

    # 1) Decide whether distillation is worth it.
    python3 scripts/distill_subgame.py diagnose \\
        blueprints/50bb_v13_pluribus --n-spots 100

    # 2) If TV > ~0.10 or argmax-disagree > ~0.20, generate targets:
    python3 scripts/distill_subgame.py distill \\
        blueprints/50bb_v13_pluribus -o /tmp/v13_distill.npz --n-spots 1000

    # 3) (Future) feed /tmp/v13_distill.npz into a boost-retrain step.

Recommended starting compute: --iterations 200 --max-deals 80, both
defaults below. For mid-sized blueprints these give ~1-3 s/spot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.deep_cfr.blueprint import Blueprint
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE
from src.solvers.subgame_distillation import SubgameDistiller


# ── Common arg parsing ───────────────────────────────────────────────────────


def _make_distiller(bp_path: str, args) -> SubgameDistiller:
    bp = Blueprint.load(bp_path, device="cpu")
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    cfr_cache = None
    if getattr(args, "cfr_cache", None):
        from src.deep_cfr.cfr_cache import CFRCache
        cfr_cache = CFRCache.load(args.cfr_cache)
        print(f"Loaded CFR cache: {len(cfr_cache)} entries from {args.cfr_cache}")
    encoder = NLHEEncoder(
        starting_stack=bp.metadata.starting_stack,
        raise_fractions=_rfs,
        cfr_cache=cfr_cache,
    )
    if bp.metadata.state_size != encoder.state_size():
        sys.exit(
            f"[error] Blueprint state_size={bp.metadata.state_size} != "
            f"encoder state_size={encoder.state_size()}. The blueprint was "
            f"trained against a different encoder shape — retrain or load a "
            f"compatible blueprint (49-dim → pass --cfr-cache)."
        )
    scenarios = None
    if args.scenarios:
        scenarios = tuple(s.strip() for s in args.scenarios.split(",")
                          if s.strip())
    return SubgameDistiller(
        blueprint=bp,
        encoder=encoder,
        base_game=game,
        iterations=args.iterations,
        max_deals=args.max_deals,
        max_walk_depth=args.max_walk_depth,
        scenarios=scenarios,
    )


# ── Subcommand: diagnose ─────────────────────────────────────────────────────


def cmd_diagnose(args) -> int:
    distiller = _make_distiller(args.blueprint, args)
    print(f"Diagnosing blueprint {args.blueprint}")
    print(f"  n_spots={args.n_spots}, CFR_iter={args.iterations}, "
          f"max_deals={args.max_deals}, max_walk_depth={args.max_walk_depth}")
    t0 = time.time()
    rep = distiller.measure_blueprint_gap(
        n_spots=args.n_spots, seed=args.seed, verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print()
    print("=" * 64)
    print(repr(rep))
    print("=" * 64)

    # Per-spot timing on the spots that actually solved (not on skipped ones).
    solved          = max(rep.n_spots, 1)
    sec_per_solve   = elapsed / solved
    skip_rate       = rep.skipped_terminal / max(rep.skipped_terminal + rep.n_spots, 1)
    print(f"elapsed: {elapsed:.1f}s "
          f"({sec_per_solve:.2f} s/solved-spot, "
          f"skip rate {skip_rate*100:.0f}%)")

    # Decision: is distillation worth running?
    gap_large = rep.mean_tv > 0.10 or rep.mean_argmax_disagreement > 0.20
    if gap_large:
        print("\n→ Gap is large: distillation is worth running.")
    else:
        print("\n→ Gap is small: blueprint is already close to locally optimal.")
        print("  Distillation will probably yield little. Focus elsewhere.")
        return 0

    # Recommend distill parameters tuned to OBSERVED throughput.
    target_budget_minutes = args.target_budget_minutes
    print(f"\n— Tuning recommendation for ~{target_budget_minutes:.0f} min "
          f"distillation budget —")

    iters     = args.iterations
    max_deals = args.max_deals
    walk      = args.max_walk_depth

    # If skip rate is high, lowering max_walk_depth reaches more non-terminal
    # roots → less wasted compute AND smaller subgames (earlier-in-the-tree
    # roots have shorter remaining play). 0.3 is a deliberate cutoff: above
    # ~30% we're wasting non-trivial throughput.
    if skip_rate > 0.30 and walk > 3:
        new_walk = max(3, walk // 2)
        print(f"  --max-walk-depth {walk} → {new_walk}  "
              f"(skip rate {skip_rate*100:.0f}%, halving cuts wasted samples + "
              f"yields smaller, faster subgames)")
        walk = new_walk

    # If avg solver time is too high, scale down iterations and/or max_deals.
    # 10 s/spot is the cutoff for "1000 spots fits in ~3 hours". Above that
    # we propose cuts proportional to overshoot.
    if sec_per_solve > 10.0:
        overshoot_iter      = max(1.0, sec_per_solve / 10.0)
        new_iters           = max(50, int(iters / overshoot_iter ** 0.5))
        new_deals           = max(20, int(max_deals / overshoot_iter ** 0.5))
        print(f"  --iterations {iters} → {new_iters}  "
              f"(observed {sec_per_solve:.0f} s/solve is "
              f"~{sec_per_solve/10:.0f}× over budget)")
        print(f"  --max-deals {max_deals} → {new_deals}  "
              f"(reduces per-iteration tree-build cost)")
        iters     = new_iters
        max_deals = new_deals

    # Estimate spots achievable at tuned parameters, conservatively assuming
    # solve-time scales linearly with iter*max_deals.
    speedup = ((args.iterations * args.max_deals) /
               max(iters * max_deals, 1))
    tuned_sec_per_solve = sec_per_solve / max(speedup, 1.0)
    # Account for skip-rate improvement from lower walk depth.
    if args.max_walk_depth != walk:
        tuned_skip_rate = max(0.0, skip_rate * (walk / args.max_walk_depth))
    else:
        tuned_skip_rate = skip_rate
    spots_per_min = 60.0 / max(tuned_sec_per_solve * (1.0 / max(1 - tuned_skip_rate, 0.05)), 1e-3)
    suggested_spots = max(50, int(spots_per_min * target_budget_minutes))
    print(f"  --n-spots {suggested_spots}  "
          f"(≈{spots_per_min:.0f} spots/min at tuned params)")

    print("\nFull suggested command:")
    print(f"  python3 {sys.argv[0]} distill {args.blueprint} \\")
    print(f"      -o targets.npz --n-spots {suggested_spots} \\")
    print(f"      --iterations {iters} --max-deals {max_deals} "
          f"--max-walk-depth {walk}")
    return 0


# ── Subcommand: distill ──────────────────────────────────────────────────────


def cmd_distill(args) -> int:
    distiller = _make_distiller(args.blueprint, args)
    print(f"Distilling subgame strategies for {args.blueprint}")
    print(f"  n_spots={args.n_spots}, CFR_iter={args.iterations}, "
          f"max_deals={args.max_deals}")

    t0 = time.time()
    targets = distiller.generate_distillation_targets(
        n_spots=args.n_spots, seed=args.seed, verbose=args.verbose,
    )
    elapsed = time.time() - t0
    if not targets:
        sys.exit("[error] No distillation targets generated.")

    # Targets are slot-indexed (length=action_size). Stack as-is — no
    # padding/aligning needed at this layer.
    action_size = len(targets[0].strategy)
    states      = np.stack([t.state for t in targets]).astype(np.float32)
    strategies  = np.stack([t.strategy for t in targets]).astype(np.float32)
    # legal_slots is variable-length per row; pad to max with -1 sentinel.
    max_legal   = max(len(t.legal_slots) for t in targets)
    legal_slots = np.full((len(targets), max_legal), -1, dtype=np.int32)
    for i, t in enumerate(targets):
        legal_slots[i, :len(t.legal_slots)] = t.legal_slots

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=states,
        strategies=strategies,
        legal_slots=legal_slots,
        blueprint_path=str(Path(args.blueprint).resolve()),
        action_size=action_size,
        n_spots=args.n_spots,
        iterations=args.iterations,
        max_deals=args.max_deals,
        format_version=2,
    )
    print(f"\n[ok] wrote {len(targets)} targets → {out}  "
          f"({states.nbytes + strategies.nbytes:,} bytes uncompressed)")
    print(f"  state.shape={states.shape}, "
          f"strategy.shape={strategies.shape} (slot-indexed, action_size={action_size}), "
          f"legal_slots.shape={legal_slots.shape}")
    print(f"  elapsed: {elapsed:.1f}s "
          f"({elapsed / max(len(targets), 1):.2f} s/target)")

    # Coverage report — flag silently narrow coverage (e.g. all 3-action
    # no-bet) before the user runs boost on a bias data set.
    print(f"\n  Coverage by legal-slot-set:")
    slot_set_counts: dict[tuple, int] = {}
    for i in range(len(targets)):
        s_set = tuple(int(s) for s in legal_slots[i] if s >= 0)
        slot_set_counts[s_set] = slot_set_counts.get(s_set, 0) + 1
    for s_set, count in sorted(slot_set_counts.items()):
        label = {
            (0,):           "check-only (degenerate)",
            (0, 1):         "face-bet fold/call (capped)",
            (0, 1, 2):      "face-bet fold/call/raise",
            (0, 1, 3):      "face-bet fold/call/allin",
            (0, 1, 2, 3):   "face-bet 4-action",
            (0, 2, 3):      "no-bet check/raise/allin",
        }.get(s_set, "unknown")
        pct = 100 * count / len(targets)
        print(f"    {str(s_set):<14s} {count:>4d} ({pct:>5.1f}%)  {label}")

    has_4action = any(len(s) == 4 for s in slot_set_counts)
    has_facebet = any(1 in s for s in slot_set_counts)
    if not has_4action:
        print(f"\n  [warn] No 4-action targets — distillation will only "
              f"boost subsets of legal action sets. Consider adding "
              f"--scenarios preflop_sb,preflop_bb_facing,flop_face_bet,"
              f"flop_no_bet for balanced coverage.")
    elif not has_facebet:
        print(f"\n  [warn] No face-bet targets — preflop and facing-bet "
              f"spots won't be boosted. Add preflop_sb,preflop_bb_facing "
              f"to --scenarios.")
    return 0


# ── Top-level CLI ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("blueprint", help="Path to blueprint directory")
    common.add_argument("--n-spots",       type=int,   default=100,
                        help="Number of sampled subgames")
    common.add_argument("--iterations",    type=int,   default=200,
                        help="Tabular CFR iterations per subgame solve")
    common.add_argument("--max-deals",     type=int,   default=80,
                        help="Subgame deal-enumeration cap")
    common.add_argument("--max-walk-depth", type=int,   default=10,
                        help="Max blueprint-self-play depth before subgame root "
                             "(only used by the legacy 'blueprint_walk' scenario).")
    common.add_argument("--scenarios",     type=str,   default="",
                        help="Comma-separated list of spot archetypes to "
                             "rotate through. Available: blueprint_walk (default, "
                             "random walk), preflop_sb, preflop_bb_facing, "
                             "flop_face_bet, flop_no_bet. Use "
                             "'preflop_sb,preflop_bb_facing,flop_face_bet,"
                             "flop_no_bet' to get balanced 4-action coverage that "
                             "the blueprint-walk sampler misses.")
    common.add_argument("--seed",          type=int,   default=0)
    common.add_argument("--cfr-cache",     type=str,   default="",
                        help="CFR advisor cache path. Required for "
                             "blueprints trained at state_size=49 "
                             "(v15_c2_v4 and later). Same cache file the "
                             "blueprint was trained against.")
    common.add_argument("--verbose", "-v", action="store_true")

    p_d = sub.add_parser("diagnose", parents=[common],
                         help="Measure blueprint vs solver disagreement")
    p_d.add_argument("--target-budget-minutes", type=float, default=60.0,
                     help="When the gap is large, the tuned-parameter "
                          "recommendation aims to fit a distillation run "
                          "into this many minutes on the observed hardware "
                          "(default 60).")
    p_d.set_defaults(func=cmd_diagnose)

    p_x = sub.add_parser("distill", parents=[common],
                         help="Generate distillation targets (.npz)")
    p_x.add_argument("-o", "--output", required=True,
                     help="Output .npz path")
    p_x.set_defaults(func=cmd_distill)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
