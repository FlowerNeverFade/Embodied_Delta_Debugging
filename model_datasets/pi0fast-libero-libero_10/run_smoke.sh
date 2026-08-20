#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

OUTPUT_DIR="$RUN_DIR/outputs/smoke" \
TASK_IDS=8 \
INIT_STATE_IDS=0 \
SEEDS=7 \
"$RUN_DIR/run_foreground.sh" --max-cases 1 "$@"
