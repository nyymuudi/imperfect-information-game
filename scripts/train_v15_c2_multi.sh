#!/usr/bin/env bash
# v15_multi_c2: multi-raise (3 sizings) + cache + aux EV.
# Source baseline: v14b_multionly (action_size=6, max_raises=1, raise_fractions=[0.33, 0.66, 1.0]).
#
# This tests whether the cache + aux-EV pipeline that worked for single-raise
# (v5_aux) generalises to the multi-raise sparse-action setting where
# Pluribus-features were memory-reported as harmful with v14d budget.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_BP="blueprints/50bb_v14b_multionly"
CACHE="blueprints/cache/v14b_multi_advisor.cache"
OUT="blueprints/50bb_v15_multi_c2"
LOG_DIR="logs"
LOG="${LOG_DIR}/v15_multi_c2_$(date +%Y%m%d_%H%M%S).log"

[[ -d "$SOURCE_BP" ]] || { echo "Missing baseline: $SOURCE_BP" >&2; exit 1; }
[[ -f "${CACHE}.npz" || -f "$CACHE" ]] || { echo "Missing cache: $CACHE (build first with build_cfr_cache.py)" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

echo "Launching v15_multi_c2 (multi-raise + cache + aux 0.1) -> $LOG"
nohup env PYTHONPATH=. python3 src/deep_cfr/train_postflop.py \
    --stack 50 --max-raises 1 \
    --raise-fractions "0.33,0.66,1.0" \
    --iterations 500 --traversals 500 --hidden 256 \
    --buffer 500000 --epochs 50 --seed 42 \
    --cfr-cache "$CACHE" \
    --aux-ev-weight 0.1 \
    --save-blueprint "$OUT" \
    --expl-games 0 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
