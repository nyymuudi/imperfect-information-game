#!/usr/bin/env python3
"""scripts/compare_ablations.py

Load N blueprints, run LBR exploitability and a CRN+EV pairwise head-to-head
on each, and print a single comparison table.

Usage:
    python3 scripts/compare_ablations.py \\
        --blueprints blueprints/50bb_v14c_v11_mimic \\
                     blueprints/50bb_v14d_new_defaults \\
                     blueprints/50bb_v14b_multionly \\
                     blueprints/50bb_v14a_no_decay \\
                     blueprints/50bb_v13_pluribus \\
        --lbr-games 600 --lbr-depth 8 --lbr-opp-samples 8 \\
        --h2h-pairs 800

What it prints:

  1. LBR table — mean ± stderr per blueprint, sorted by exploitability.
  2. Significance grid for LBR differences (z-scores between every pair).
  3. CRN+EV pairwise h2h matrix — A's mean payoff in mbb/pair when each
     pair plays both seats with shared deals + action seeds. Useful when
     LBR is noise-dominated but blueprints differ in actual play.
  4. Final summary: which blueprint is the LBR winner and whether it also
     wins h2h against the runners-up.

Both metrics are computed with the same seed across all blueprints so
relative comparisons are apples-to-apples.

Skip a metric by passing 0:
  --lbr-games 0    → skip LBR (use when measurement is too slow)
  --h2h-pairs 0    → skip h2h (LBR-only mode)
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.exploitability import estimate_exploitability
from src.analysis.head_to_head import match_crn
from src.deep_cfr.blueprint import Blueprint
from src.deep_cfr.state_encoder import NLHEEncoder
from src.games.postflop_nlhe import PostflopNLHE


def _short(path: str) -> str:
    """Trim a blueprint path to a printable identifier."""
    return Path(path).name


def _load_bundle(path: str, cfr_cache=None) -> tuple:
    """Return (blueprint, game, encoder) configured to match the saved metadata.

    The game and encoder are reconstructed from blueprint metadata so each
    blueprint runs on its own action space and raise_fractions tuple — a
    multi-raise blueprint will not be silently played on a 4-action game.

    When ``cfr_cache`` is provided, the encoder is built with it attached.
    The blueprint's metadata state_size must match the cache-augmented
    encoder's state_size (49 instead of 37) — load with the SAME cache
    file the blueprint was trained against.
    """
    bp = Blueprint.load(path, device="cpu")
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    # Per-blueprint cache attachment: only attach if the blueprint was
    # trained at 49 dims (state_size matches cache-augmented encoder).
    # Mixing a cache-only blueprint with a no-cache one is supported by
    # keying each side to its own encoder downstream.
    use_cache = cfr_cache is not None and bp.metadata.state_size == 49
    encoder = NLHEEncoder(
        starting_stack=bp.metadata.starting_stack,
        raise_fractions=_rfs,
        cfr_cache=cfr_cache if use_cache else None,
    )
    if bp.metadata.state_size != encoder.state_size():
        sys.exit(
            f"[error] {path}: state_size {bp.metadata.state_size} "
            f"≠ encoder {encoder.state_size()} — re-encode incompatibility "
            f"(pass --cfr-cache when comparing 49-dim blueprints)"
        )
    return bp, game, encoder


# ── LBR pass ────────────────────────────────────────────────────────────────


def _lbr_pass(bps_and_games, n_games: int, n_opp: int, max_depth: int,
              seed: int, verbose: bool) -> dict[str, object]:
    print("=" * 72)
    print(f"LBR exploitability (n_games={n_games}, n_opp_samples={n_opp}, "
          f"max_depth={max_depth}, seed={seed})")
    print("=" * 72)
    results = {}
    for label, (bp, game, encoder) in bps_and_games.items():
        t0 = time.time()
        r = estimate_exploitability(
            bp, game, encoder,
            n_games=n_games, n_opp_samples=n_opp,
            max_node_depth=max_depth, seed=seed,
            verbose=verbose,
        )
        elapsed = time.time() - t0
        print(f"  {label:<32s}: {float(r):8.1f} ± {r.stderr_mbb:6.1f} "
              f"mbb/dec  (n={r.n_decisions:>4d}, t={elapsed:.0f}s)")
        results[label] = r
    return results


def _print_lbr_significance(lbr: dict, threshold_z: float = 1.96) -> None:
    if len(lbr) < 2:
        return
    print("\n— LBR pairwise z-scores (|z| ≥ 1.96 → 95% sig) —")
    labels = list(lbr.keys())
    # Header
    print(f"  {'':<24s} " + " ".join(f"{l[:14]:>14s}" for l in labels))
    for a in labels:
        row = [f"  {a[:24]:<24s} "]
        for b in labels:
            if a == b:
                row.append(f"{'·':>14s}")
                continue
            mean_diff = float(lbr[a]) - float(lbr[b])
            stderr_combined = (lbr[a].stderr_mbb ** 2 + lbr[b].stderr_mbb ** 2) ** 0.5
            z = mean_diff / max(stderr_combined, 1e-9)
            marker = "*" if abs(z) >= threshold_z else " "
            row.append(f"{z:>+13.2f}{marker}")
        print(" ".join(row))
    print("  (rows = A, cols = B; positive z → A has HIGHER LBR than B "
          "(worse, more exploitable))")


# ── CRN+EV h2h pass ─────────────────────────────────────────────────────────


def _h2h_pass(bps_and_games, n_pairs: int, seed: int) -> dict:
    print("\n" + "=" * 72)
    print(f"CRN+EV head-to-head (n_pairs={n_pairs}, seed={seed})")
    print("=" * 72)

    labels = list(bps_and_games.keys())
    # NxN matrix of mean_diff_mbb_per_pair where entry [a][b] = A's payoff
    # when A and B alternate seats over shared deals.
    matrix = {a: {b: None for b in labels} for a in labels}

    for a, b in combinations(labels, 2):
        bp_a, game_a, enc_a = bps_and_games[a]
        bp_b, game_b, enc_b = bps_and_games[b]
        # h2h requires both blueprints to share an action space. If they
        # differ, the result is meaningless — skip and report.
        if game_a.raise_fractions != game_b.raise_fractions:
            print(f"  {a} vs {b}: SKIPPED (different action spaces: "
                  f"{game_a.raise_fractions} vs {game_b.raise_fractions})")
            continue
        t0 = time.time()
        r = match_crn(bp_a, bp_b, game_a, enc_a, encoder_b=enc_b,
                      n_pairs=n_pairs, seed=seed, ev_adjusted=True)
        elapsed = time.time() - t0
        print(f"  {a[:30]:<30s} vs {b[:30]:<30s}: "
              f"A={r['win_rate_mbb']:+8.1f} ± {r['stderr_mbb']:5.1f} "
              f"mbb/pair  (t={elapsed:.0f}s)")
        matrix[a][b] =  r["win_rate_mbb"]
        matrix[b][a] = -r["win_rate_mbb"]
    return matrix


def _print_h2h_table(matrix: dict, labels: list[str]) -> None:
    print("\n— h2h pairwise table (A's mean mbb/pair vs B, positive → A wins) —")
    print(f"  {'':<24s} " + " ".join(f"{l[:14]:>14s}" for l in labels))
    for a in labels:
        row = [f"  {a[:24]:<24s} "]
        for b in labels:
            v = matrix[a][b]
            if a == b:
                row.append(f"{'·':>14s}")
            elif v is None:
                row.append(f"{'incomp':>14s}")
            else:
                row.append(f"{v:>+13.0f} ")
        print(" ".join(row))


# ── Summary ─────────────────────────────────────────────────────────────────


def _final_summary(lbr: dict, h2h: dict | None) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if lbr:
        ranked = sorted(lbr.items(), key=lambda kv: float(kv[1]))
        winner_label, winner_r = ranked[0]
        print(f"  LBR winner       : {winner_label}  "
              f"({float(winner_r):.1f} ± {winner_r.stderr_mbb:.1f} mbb/dec)")
        if len(ranked) > 1:
            second_label, second_r = ranked[1]
            gap   = float(second_r) - float(winner_r)
            stderr = (winner_r.stderr_mbb ** 2 + second_r.stderr_mbb ** 2) ** 0.5
            z = gap / max(stderr, 1e-9)
            print(f"  vs runner-up     : {second_label}  Δ={gap:+.1f} "
                  f"mbb/dec  (z={z:+.2f}, "
                  f"{'sig' if abs(z) >= 1.96 else 'NOT sig'} at 95%)")

    if h2h is not None and lbr:
        labels = list(lbr.keys())
        net_h2h = {a: 0.0 for a in labels}
        n_pair  = {a: 0 for a in labels}
        for a in labels:
            for b in labels:
                if a == b: continue
                v = h2h[a][b]
                if v is None: continue
                net_h2h[a] += v
                n_pair[a]  += 1
        ranked_h2h = sorted(
            net_h2h.items(),
            key=lambda kv: (kv[1] / n_pair[kv[0]]) if n_pair[kv[0]] else 0.0,
            reverse=True,
        )
        h_winner = ranked_h2h[0][0]
        avg_win = net_h2h[h_winner] / max(n_pair[h_winner], 1)
        print(f"  h2h winner       : {h_winner}  "
              f"(avg {avg_win:+.0f} mbb/pair vs others)")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--blueprints", "-b", nargs="+", required=True,
                   help="Paths to blueprint directories to compare")
    p.add_argument("--lbr-games",       type=int, default=600,
                   help="LBR n_games per blueprint (0 to skip)")
    p.add_argument("--lbr-opp-samples", type=int, default=8,
                   help="LBR opponent-card marginalisation samples")
    p.add_argument("--lbr-depth",       type=int, default=8,
                   help="LBR max_node_depth")
    p.add_argument("--h2h-pairs",       type=int, default=800,
                   help="CRN+EV head-to-head n_pairs (0 to skip)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--cfr-cache",       type=str, default="",
                   help="Optional CFR advisor cache path. Required when "
                        "loading blueprints that were trained with "
                        "--cfr-cache (their state_size includes 12 advisor "
                        "dims and reloading without the cache mismatches "
                        "shape). Pass the same .cache file the blueprint "
                        "was trained against.")
    p.add_argument("--verbose", "-v",   action="store_true")
    args = p.parse_args()

    # Validate paths
    missing = [bp for bp in args.blueprints if not Path(bp).exists()]
    if missing:
        sys.exit(f"[error] Missing blueprints: {missing}")

    cfr_cache = None
    if args.cfr_cache:
        from src.deep_cfr.cfr_cache import CFRCache
        cfr_cache = CFRCache.load(args.cfr_cache)
        print(f"Loaded CFR cache: {len(cfr_cache)} entries from {args.cfr_cache}")

    # Load all blueprints up-front so any incompatibility surfaces early.
    print(f"Loading {len(args.blueprints)} blueprints...")
    bps_and_games = {}
    for bp_path in args.blueprints:
        label = _short(bp_path)
        if label in bps_and_games:
            sys.exit(f"[error] Duplicate label after shortening: {label}")
        bps_and_games[label] = _load_bundle(bp_path, cfr_cache=cfr_cache)
        print(f"  loaded {label}")
    print()

    lbr_results: dict = {}
    if args.lbr_games > 0:
        lbr_results = _lbr_pass(
            bps_and_games, args.lbr_games, args.lbr_opp_samples,
            args.lbr_depth, args.seed, args.verbose,
        )
        _print_lbr_significance(lbr_results)
    else:
        print("(LBR skipped: --lbr-games=0)")

    h2h_matrix = None
    if args.h2h_pairs > 0:
        h2h_matrix = _h2h_pass(bps_and_games, args.h2h_pairs, args.seed)
        _print_h2h_table(h2h_matrix, list(bps_and_games.keys()))
    else:
        print("(h2h skipped: --h2h-pairs=0)")

    _final_summary(lbr_results, h2h_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
