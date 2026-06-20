#!/usr/bin/env bash
# Train v15_c2: v14d_fixed + CFR advisor cache (state_size 37 -> 49)
set -euo pipefail

cd "$(dirname "$0")/.."

CACHE="blueprints/cache/v14d_advisor_v4.cache"
OUT="blueprints/50bb_v15_c2_v4"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v4_$(date +%Y%m%d_%H%M%S).log"

[[ -f "$CACHE" || -f "${CACHE}.npz" ]] || { echo "Missing cache: $CACHE(.npz)" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2 training -> $LOG"
echo "  cache:  $CACHE"
echo "  output: $OUT"

nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &

PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
echo "Kill: kill $PID"
