#!/usr/bin/env bash
# v15_c2_v7_a{X}: aux α-sweep (Path B). Same v5_aux config, just different α.
# Usage: ./scripts/train_v15_c2_aux_sweep.sh ALPHA  (e.g. 0.05, 0.20, 0.30)
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHA="${1:-0.05}"
ALPHA_TAG="$(echo "$ALPHA" | tr '.' 'p')"

CACHE="blueprints/cache/v14d_advisor_v4.cache"
OUT="blueprints/50bb_v15_c2_v7_a${ALPHA_TAG}"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v7_a${ALPHA_TAG}_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v7_a${ALPHA_TAG} (aux=$ALPHA) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" \
    --aux-ev-weight "$ALPHA" \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID, alpha=$ALPHA"
echo "Tail: tail -f $LOG"
