#!/usr/bin/env bash
# G:n esivaihe — Cache multi-raise blueprintille (v14b_multionly).
# Action_size 6, 3 raise-kokoa: 0.33, 0.66, 1.0.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_BP="blueprints/50bb_v14b_multionly"
OUT="blueprints/cache/v14b_multi_advisor.cache"
LOG_DIR="logs"
LOG="${LOG_DIR}/cache_v14b_multi_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$(dirname "$OUT")"
echo "Building multi-raise cache from $SOURCE_BP → $OUT"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 scripts/build_cfr_cache.py \
    "$SOURCE_BP" \
    -o "$OUT" \
    --n-trajectories 80000 --n-spots 3000 \
    --iter-per-spot 30 --max-deals 15 \
    --workers 4 -v \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
