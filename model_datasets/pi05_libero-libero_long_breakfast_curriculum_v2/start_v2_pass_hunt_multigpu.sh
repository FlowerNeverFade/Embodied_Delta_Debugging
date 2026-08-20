#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
OUTPUT_DIR="${V2_PASS_HUNT_OUTPUT_DIR:-$RUN_DIR/outputs/v2_pass_hunt_20260526}"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/master.log"
nohup setsid bash -c "cd '$PROJECT_ROOT' && exec /root/autodl-tmp/envs/libero38/bin/python -u '$RUN_DIR/v2_pass_hunt_multigpu.py' --output-dir '$OUTPUT_DIR' ${V2_PASS_HUNT_ARGS:-}" \
  >> "$OUTPUT_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR/run.pid"
echo "started pid=$(cat "$OUTPUT_DIR/run.pid")"
echo "log=$OUTPUT_DIR/master.log"
echo "summary=$OUTPUT_DIR/summary.json"
echo "manifest=$OUTPUT_DIR/manifest.jsonl"
