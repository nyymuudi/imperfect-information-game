#!/usr/bin/env bash
# D — v15_c2_v8_pf: same config as v5_aux but preflop-rich cache (2776 spots, iter=80).
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="blueprints/cache/v14d_advisor_v5_pf.cache"
OUT="blueprints/50bb_v15_c2_v8_pf"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v8_pf_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v8_pf (preflop-rich cache, aux=0.1) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" --aux-ev-weight 0.1 \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
