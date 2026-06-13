#!/usr/bin/env python3
"""
Parse historical validation_runs/*.log files and create retroactive MLflow runs.

Reads per-iter trace (regex on "iter   N: regret_buf=... loss=... expl=... t=...s")
and the final "Final blueprint exploitability: X mbb/decision" line. Derives
params from the header block. Pure log replay — does not retrain anything.

Usage:
    python3 scripts/backfill_mlflow.py [--logs-dir validation_runs/]
"""

import argparse
import re
import sys
from pathlib import Path

# Hard-coded mapping from log basename → (run_name, bucket_scheme, notes).
# These reflect the ablation history documented in the user's CLAUDE.md +
# session notes (see also validation_runs/*.log headers).
RUN_METADATA = {
    "run_20260607_1445.log":          ("v2_first_run",          "subcat-finequant", "v2 baseline pre-coarse-quant"),
    "ablation_nofinetune_20260607_1811.log":
                                      ("ablation_A_no_finetune","subcat-finequant", "LR decay on, no fine-tune"),
    "ablation_nolrdecay_20260607_2207.log":
                                      ("ablation_B_no_lrdecay", "subcat-finequant", "LR decay off, no fine-tune"),
    "ablation_subcat_manual.log":     ("v3_coarse",             "flat-K8",          "v3 production winner"),
    "v3_coarse.log":                  ("v3_coarse_rerun",       "flat-K8",          "v3 rerun"),
    "v4_k16.log":                     ("v4_k16",                "flat-K16",         "K=16 ablation"),
    "v5_tree.log":                    ("v5_tree_4x4",           "tree-4x4",         "tree super+fine 4x4"),
    "v6_super.log":                   ("v6_super_K4",           "super-K4",         "super-only K=4"),
    "v7_tree42.log":                  ("v7_tree42_4x2",         "tree42-4x2",       "low/high half-split"),
    "v8_mcehs.log":                   ("v8_mc_ehs",             "flat-K8",          "MC EHS continuous board feature"),
    "v9_evmccfr.log":                 ("v9_ev_mccfr",           "flat-K8",          "EV-adjusted MCCFR terminals"),
}

ITER_RE = re.compile(
    r"iter\s+(\d+):\s+regret_buf=\s*([\d,]+)\s*\|\s*"
    r"strat_buf=\s*([\d,]+)\s*\|\s*loss=([\d.]+)"
    r"\s+expl=\s*([\d.]+|\s*n/a)\s*mbb/decision\s*\|\s*t=([\d.]+)s"
)
FINAL_RE  = re.compile(r"Final blueprint exploitability:\s*([\d.]+)\s+mbb/decision")
HEADER_STACK = re.compile(r"Stack:\s+(\d+)BB")
HEADER_ITER  = re.compile(r"Iterations:\s+(\d+)")
HEADER_HIDDEN = re.compile(r"Hidden:\s+(\d+)")
HEADER_BUF   = re.compile(r"Regret buf:\s+([\d,]+)")
SCHEME_RE    = re.compile(r"Encoder scheme:\s+(\w+)")
RAISE_RE     = re.compile(r"Raise:\s+(\d+)% pot")


def parse_log(path: Path) -> dict | None:
    """Return dict of {params, iter_metrics, final_expl} or None on parse failure."""
    text = path.read_text(errors="ignore")
    if not text.strip():
        return None
    params = {"_source_log": str(path)}
    if m := HEADER_STACK.search(text):  params["stack"] = int(m.group(1))
    if m := HEADER_ITER.search(text):   params["iterations"] = int(m.group(1))
    if m := HEADER_HIDDEN.search(text): params["hidden"] = int(m.group(1))
    if m := HEADER_BUF.search(text):    params["buffer"] = int(m.group(1).replace(",",""))
    if m := SCHEME_RE.search(text):     params["scheme_logged"] = m.group(1)
    if m := RAISE_RE.search(text):      params["raise_fraction"] = int(m.group(1)) / 100.0

    iter_metrics = []
    for m in ITER_RE.finditer(text):
        i, rbuf, sbuf, loss, expl_str, t = m.groups()
        iter_metrics.append({
            "step": int(i),
            "regret_buf_size":   int(rbuf.replace(",", "")),
            "strategy_buf_size": int(sbuf.replace(",", "")),
            "regret_loss":       float(loss),
            "expl_current_strategy":
                None if expl_str.strip() == "n/a" else float(expl_str),
            "elapsed_sec":       float(t),
        })

    final_expl = None
    if m := FINAL_RE.search(text):
        final_expl = float(m.group(1))

    if not iter_metrics and final_expl is None:
        return None    # nothing meaningful in the log

    return {"params": params, "iters": iter_metrics, "final_expl": final_expl}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="validation_runs",
                   help="Directory with *.log files to backfill.")
    p.add_argument("--experiment", default="deep-cfr-nlhe",
                   help="MLflow experiment name (default: deep-cfr-nlhe).")
    args = p.parse_args()

    import mlflow
    tracking_db = (Path.cwd() / "mlflow.db").resolve()
    artifacts   = (Path.cwd() / "mlruns").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    try:
        exp = mlflow.get_experiment_by_name(args.experiment)
        if exp is None:
            mlflow.create_experiment(args.experiment,
                                      artifact_location=f"file://{artifacts}")
    except Exception:
        pass
    mlflow.set_experiment(args.experiment)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"No such dir: {logs_dir}", file=sys.stderr)
        return 1

    written = 0
    skipped = 0
    for log_path in sorted(logs_dir.glob("*.log")):
        meta = RUN_METADATA.get(log_path.name)
        if meta is None:
            print(f"  skip (no metadata): {log_path.name}")
            skipped += 1
            continue
        run_name, bucket_scheme, note = meta
        parsed = parse_log(log_path)
        if parsed is None:
            print(f"  skip (empty/unparseable): {log_path.name}")
            skipped += 1
            continue

        with mlflow.start_run(run_name=run_name):
            # Params from header.
            for k, v in parsed["params"].items():
                mlflow.log_param(k, v)
            mlflow.log_param("bucket_scheme_inferred", bucket_scheme)
            mlflow.log_param("note", note)
            mlflow.log_param("backfilled", True)
            # Per-iter metrics.
            for m in parsed["iters"]:
                step = m["step"]
                for key, val in m.items():
                    if key == "step" or val is None:
                        continue
                    mlflow.log_metric(key, float(val), step=step)
            # Final expl.
            if parsed["final_expl"] is not None:
                mlflow.log_metric("final_exploitability_mbb_per_decision",
                                   float(parsed["final_expl"]))
            # Source log as artifact.
            mlflow.log_artifact(str(log_path))
        print(f"  ✓ {run_name}  ({log_path.name})  "
              f"final={parsed['final_expl']}  iters_logged={len(parsed['iters'])}")
        written += 1

    print(f"\nDone: {written} backfilled, {skipped} skipped.")
    print(f"View: mlflow ui --backend-store-uri sqlite:///{tracking_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
