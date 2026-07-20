#!/usr/bin/env python3
"""
Deploy a trained blueprint to the web frontend.

Copies strategy_net.onnx + versioned .onnx + model_manifest.json from
blueprints/<name>/ to web/public/models/. The web app picks up the new
version automatically on next page load (cache-busted by the hashed
filename in the manifest).

Usage:
    python3 scripts/deploy_to_web.py blueprints/50bb_v7_long_production

Optional:
    --dry-run         show what would be copied without writing
    --clean           remove any other strategy_net.*.onnx in web/public/models/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("blueprint", help="Blueprint directory (e.g. blueprints/50bb_v7_long_production)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show actions without writing.")
    p.add_argument("--clean", action="store_true",
                   help="Remove any other strategy_net.*.onnx versions first.")
    p.add_argument("--web-models",
                   default="web/public/models",
                   help="Target web models dir (default: web/public/models).")
    p.add_argument("--cache", default=None,
                   help="Advisor cache .bin to deploy alongside the model "
                        "(sets manifest cache_path/cache_entries).")
    p.add_argument("--lbr", type=float, default=None,
                   help="LBR number for the manifest (mbb/decision).")
    p.add_argument("--lbr-note", default=None,
                   help="Metric/provenance note for the LBR number, e.g. "
                        "'LBR v2 bayes+stratified, 5 seeds'.")
    p.add_argument("--action-size", type=int, default=None,
                   help="Network output slots (manifest action_size; the "
                        "web builds masks from this — 4 legacy, 6 multi-raise).")
    p.add_argument("--legal-slots", default=None,
                   help="Comma list of 0/1 per slot, e.g. '1,1,1,1,0,1' "
                        "for a 2-size tree (RAISE_2 slot untrained).")
    args = p.parse_args()

    bp_dir = Path(args.blueprint)
    if not bp_dir.is_dir():
        print(f"Error: {bp_dir} is not a directory", file=sys.stderr)
        return 1

    manifest_src = bp_dir / "model_manifest.json"
    onnx_src     = bp_dir / "strategy_net.onnx"

    for f in (manifest_src, onnx_src):
        if not f.exists():
            print(f"Error: {f} missing — blueprint not fully exported", file=sys.stderr)
            return 1

    manifest = json.loads(manifest_src.read_text())
    versioned_name = manifest["model_path"].split("/")[-1]
    versioned_src  = bp_dir / versioned_name
    if not versioned_src.exists():
        print(f"Error: versioned ONNX {versioned_src} missing", file=sys.stderr)
        return 1

    web_models = Path(args.web_models)
    if not web_models.exists():
        if args.dry_run:
            print(f"  [dry-run] mkdir {web_models}")
        else:
            web_models.mkdir(parents=True, exist_ok=True)

    plan = [
        (onnx_src,      web_models / "strategy_net.onnx"),
        (versioned_src, web_models / versioned_name),
    ]

    # Manifest is written (not copied) so deploy-time fields can be merged:
    # cache pointer, published LBR number + metric note, action layout.
    manifest_out = dict(manifest)
    manifest_out["blueprint"] = bp_dir.name
    if args.cache:
        cache_src = Path(args.cache)
        if not cache_src.exists():
            print(f"Error: cache {cache_src} missing", file=sys.stderr)
            return 1
        plan.append((cache_src, web_models / cache_src.name))
        manifest_out["cache_path"] = f"/models/{cache_src.name}"
        import struct
        with open(cache_src, "rb") as f:
            hdr = f.read(12)
        manifest_out["cache_entries"] = struct.unpack("<III", hdr)[2]
    if args.lbr is not None:
        manifest_out["lbr_mbb_per_decision"] = args.lbr
    if args.lbr_note:
        manifest_out["lbr_note"] = args.lbr_note
    if args.action_size is not None:
        manifest_out["action_size"] = args.action_size
    if args.legal_slots:
        manifest_out["legal_slots"] = [int(x) for x in
                                       args.legal_slots.split(",")]

    print(f"Blueprint: {bp_dir}")
    print(f"  iterations:  {manifest.get('iterations')}")
    print(f"  state_size:  {manifest.get('state_size')}")
    print(f"  hash:        {manifest.get('hash')}")
    print(f"  → {web_models}/")
    print()

    if args.clean:
        for old in web_models.glob("strategy_net.*.onnx"):
            if old.name == versioned_name:
                continue
            if args.dry_run:
                print(f"  [dry-run] rm  {old}")
            else:
                old.unlink()
                print(f"  rm  {old.name}")

    for src, dst in plan:
        if args.dry_run:
            print(f"  [dry-run] cp  {src} → {dst}")
        else:
            # Remove any existing dst first: a legacy strategy_net.onnx was a
            # symlink to a versioned file, and copy2 would follow it and
            # overwrite the WRONG (linked) target instead of replacing the
            # link. Unlinking makes every deploy write a real, independent file.
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)
            print(f"  cp  {dst.name}")

    manifest_dst = web_models / "model_manifest.json"
    if args.dry_run:
        print(f"  [dry-run] write {manifest_dst}: {json.dumps(manifest_out)}")
    else:
        manifest_dst.write_text(json.dumps(manifest_out, indent=2) + "\n")
        print(f"  write {manifest_dst.name}")

    print()
    if args.dry_run:
        print("Dry run — no files written. Drop --dry-run to deploy.")
    else:
        print("✓ Deploy complete.")
        print(f"  Reload web frontend to pick up the new model.")
        print(f"  (Content-hashed filename auto-busts browser cache.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
