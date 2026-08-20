#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${V2_PASS_HUNT_OUTPUT_DIR:-$RUN_DIR/outputs/v2_pass_hunt_20260526}"

if [[ -f "$OUTPUT_DIR/run.pid" ]]; then
  pid="$(cat "$OUTPUT_DIR/run.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping runner pid=$pid"
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  fi
fi

pkill -f "v2_pass_hunt_multigpu.py" 2>/dev/null || true
pkill -f "pi05_natural_failure_probe.py.*v2_pass_hunt_20260526" 2>/dev/null || true
pkill -f "scripts/serve_policy.py.*--port 806" 2>/dev/null || true
pkill -f "serve_lerobot_pi0fast_policy.py.*--port 806" 2>/dev/null || true

echo "paused output=$OUTPUT_DIR"
