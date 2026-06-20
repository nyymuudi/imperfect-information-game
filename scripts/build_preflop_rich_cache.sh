#!/usr/bin/env bash
# D — Preflop-rich cache: 5000 spots, iter_per_spot=80 (vs v4:n 2000/30).
# Mining vahvisti että preflop SB-open + BB-facing-raise vuotavat eniten
# exploitabilityä; lisäbudjetti niihin advisor-signaalia tarkentamaan.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_BP="blueprints/50bb_v14d_fixed"
OUT="blueprints/cache/v14d_advisor_v5_pf.cache"
LOG_DIR="logs"
LOG="${LOG_DIR}/cache_v5_pf_build_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$(dirname "$OUT")"
echo "Building preflop-rich cache → $OUT (log $LOG)"
echo "  spots=5000, iter=80, max_deals=20, vs-baseline, 4 workers"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 scripts/build_cfr_cache.py \
    "$SOURCE_BP" \
    -o "$OUT" \
    --n-trajectories 80000 --n-spots 5000 \
    --iter-per-spot 80 --max-deals 20 \
    --workers 4 -v \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
