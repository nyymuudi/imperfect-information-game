#!/usr/bin/env bash
# v15_c2_v10_aug: same config as v5_aux + counterfactual hand-bucket
# augmentation. Forces network to be smooth across equity-bucket abstraction.
set -euo pipefail
cd "$(dirname "$0")/.."

PROB="${1:-0.3}"
PROB_TAG="$(echo "$PROB" | tr '.' 'p')"

CACHE="blueprints/cache/v14d_advisor_v4.cache"
OUT="blueprints/50bb_v15_c2_v10_aug_${PROB_TAG}"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v10_aug_${PROB_TAG}_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v10_aug_${PROB_TAG} (bucket-aug prob=${PROB}) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" --aux-ev-weight 0.1 \
    --aug-bucket-prob "$PROB" --aug-bucket-radius 1 \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
