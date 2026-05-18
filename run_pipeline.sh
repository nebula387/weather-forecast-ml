#!/usr/bin/env bash
# run_pipeline.sh — fetch live data and generate predictions (cron-ready)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$TIMESTAMP] Pipeline started"

python src/fetch_live.py || {
    echo "[$TIMESTAMP] ERROR: fetch_live.py failed" >&2
    exit 1
}

python src/predict.py || {
    echo "[$TIMESTAMP] ERROR: predict.py failed" >&2
    exit 1
}

echo "[$TIMESTAMP] Pipeline completed successfully"
