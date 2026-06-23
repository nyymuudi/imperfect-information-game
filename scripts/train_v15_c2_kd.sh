#!/usr/bin/env bash
# v15_c2_v9_kd: same config as v5_aux but with KL-distillation from v5_aux
# teacher during strategy-network training.
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="blueprints/cache/v14d_advisor_v4.cache"
TEACHER="blueprints/50bb_v15_c2_v5_aux"
KL_WEIGHT="${1:-0.5}"
KL_TAG="$(echo "$KL_WEIGHT" | tr '.' 'p')"
OUT="blueprints/50bb_v15_c2_v9_kd_${KL_TAG}"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v9_kd_${KL_TAG}_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
[[ -d "$TEACHER" ]] || { echo "Missing teacher: $TEACHER" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v9_kd_${KL_TAG} (teacher=v5_aux, KL=${KL_WEIGHT}) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" --aux-ev-weight 0.1 \
    --teacher-blueprint "$TEACHER" --teacher-kl-weight "$KL_WEIGHT" \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID, KL=${KL_WEIGHT}"
echo "Tail: tail -f $LOG"
