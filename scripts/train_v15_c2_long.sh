#!/usr/bin/env bash
# v15_c2_v4 LONG: 1000 iter, hidden 512 — push the trained-pipeline ceiling.
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="blueprints/cache/v14d_advisor_v4.cache"
OUT="blueprints/50bb_v15_c2_v4_long"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v4_long_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v4_long (1000 iter, hidden 512) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 1000 --traversals 500 --hidden 512 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
