#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
if [[ -f "$OUTPUT_DIR/run.pid" ]]; then
  kill "$(cat "$OUTPUT_DIR/run.pid")" 2>/dev/null || true
fi
pkill -f "serve_policy.py --port $POLICY_PORT" 2>/dev/null || true
pkill -f "pi05_natural_failure_probe.py --policy-port $POLICY_PORT" 2>/dev/null || true
echo "stopped $MODEL_NAME-$DATASET_NAME"
