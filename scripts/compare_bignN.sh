#!/usr/bin/env bash
# E — Big-N compare: 1200 LBR-pelejä, 2000 h2h-paria.
# Stderr puolittuu vs default 400/500 → todelliset signaalit erottuvat
# kohinasta. Käyttöesimerkki: ./scripts/compare_bignN.sh BP1 BP2 [BP3 ...]
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="blueprints/cache/v14d_advisor_v4.cache"
LOG_DIR="logs"
LOG="${LOG_DIR}/compare_bignN_$(date +%Y%m%d_%H%M%S).log"

[[ $# -ge 2 ]] || { echo "Usage: $0 BP1 BP2 [BP3 ...]" >&2; exit 1; }
mkdir -p "$LOG_DIR"

echo "Launching big-N compare → $LOG"
echo "  blueprints: $*"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 scripts/compare_ablations.py \
    --blueprints "$@" \
    --cfr-cache "$CACHE" \
    --lbr-games 1200 --h2h-pairs 2000 \
    > "$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Tail: tail -f $LOG"
