#!/usr/bin/env bash
# v15_c2_v11_pcfr: same config as v5_aux + Predictive CFR+ (Brown 2020).
# Adds momentum alpha*(r_t - r_{t-1}) to each CFR+ accumulator update.
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHA="${1:-1.0}"
ALPHA_TAG="$(echo "$ALPHA" | tr '.' 'p')"

CACHE="blueprints/cache/v14d_advisor_v4.cache"
OUT="blueprints/50bb_v15_c2_v11_pcfr_${ALPHA_TAG}"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_c2_v11_pcfr_${ALPHA_TAG}_$(date +%Y%m%d_%H%M%S).log"

[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_c2_v11_pcfr_${ALPHA_TAG} (Predictive CFR+ alpha=${ALPHA}) -> $LOG"
# Note: regret_target must be cfrplus for predictive_alpha to take effect.
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 --raise-fraction 0.5 \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --regret-target cfrplus \
    --cfr-cache "$CACHE" --aux-ev-weight 0.1 \
    --predictive-alpha "$ALPHA" \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
