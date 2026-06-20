#!/usr/bin/env python3
"""scripts/inspect_gap_map.py

Decode + summarise an exploit-gap .npz from mine_exploit_gaps.py.
Outputs ASCII histogram + top-N worst spots with human-readable
abstraction-field decoding.

Usage:
    python3 scripts/inspect_gap_map.py PATH.npz [--top 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.deep_cfr.cfr_cache import _FIELDS, _OFFSETS


_STREET = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
_LA_REL = {0: "no-aggro", 1: "me", 2: "opp"}


def decode_key(k: int) -> dict:
    out = {}
    for name, width in _FIELDS:
        out[name] = (k >> _OFFSETS[name]) & ((1 << width) - 1)
    return out


def describe(k: int) -> str:
    f = decode_key(k)
    return (
        f"{_STREET.get(f['street'], '?'):7s} "
        f"P{f['player']} "
        f"raises={f['raises']} "
        f"la={_LA_REL.get(f['last_aggressor'], '?'):8s} "
        f"pot_b={f['pot_bucket']} "
        f"spr_b={f['spr_bucket']} "
        f"board_b={f['board_bucket']} "
        f"hand_b={f['hand_bucket']}"
    )


def ascii_hist(values: np.ndarray, n_bins: int = 20, width: int = 60) -> str:
    if len(values) == 0:
        return "(empty)"
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    mx = counts.max()
    lines = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(width * counts[i] / max(mx, 1))
        lines.append(
            f"  {lo:8.0f} .. {hi:8.0f} | "
            f"{'#' * bar_len:<{width}s} {counts[i]:>6d}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to gap-map .npz")
    p.add_argument("--top", type=int, default=15,
                   help="Show top-N worst spots")
    args = p.parse_args()

    if not Path(args.path).exists():
        sys.exit(f"[error] not found: {args.path}")

    d = np.load(args.path, allow_pickle=True)
    keys   = np.asarray(d["keys"],   dtype=np.uint64)
    gaps   = np.asarray(d["gaps"],   dtype=np.float32)
    visits = np.asarray(d["visits"], dtype=np.int32)
    total  = int(d.get("total_decisions", -1))

    print(f"Gap map: {args.path}")
    print(f"  blueprint: {d.get('blueprint_path', '?')}")
    print(f"  built at:  {d.get('timestamp', '?')}")
    print(f"  N unique keys: {len(keys)}")
    print(f"  total decisions sampled: {total}")
    print(f"  avg visits per key: {visits.mean():.1f}  (median {np.median(visits):.0f})")
    print()
    print(f"  gap mbb stats:")
    print(f"    min:    {gaps.min():.1f}")
    print(f"    25%:    {np.percentile(gaps, 25):.1f}")
    print(f"    median: {np.median(gaps):.1f}")
    print(f"    mean:   {gaps.mean():.1f}")
    print(f"    75%:    {np.percentile(gaps, 75):.1f}")
    print(f"    95%:    {np.percentile(gaps, 95):.1f}")
    print(f"    max:    {gaps.max():.1f}")

    print(f"\nHistogram (mbb):")
    print(ascii_hist(gaps))

    # Top-N worst spots, weighted by gap (high gap + lots of visits = most loss)
    weight = gaps * visits.astype(np.float32)
    order  = np.argsort(weight)[::-1][: args.top]

    print(f"\nTop {args.top} most-exploited spots (weighted by gap × visits):")
    print(f"  {'gap_mbb':>9s} {'visits':>7s}  {'spot description':s}")
    for i in order:
        print(f"  {gaps[i]:9.1f} {visits[i]:7d}  {describe(int(keys[i]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
