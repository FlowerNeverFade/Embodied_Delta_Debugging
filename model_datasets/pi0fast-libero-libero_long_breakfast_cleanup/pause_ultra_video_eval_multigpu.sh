#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_long_breakfast_video_causal_v2_multigpu_20260526}"

for pid_file in "$BASE_OUTPUT_DIR"/gpu*/run.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  echo "stopping runner pid=$pid from $pid_file"
  pkill -TERM -s "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
done

sleep 5
for pid_file in "$BASE_OUTPUT_DIR"/gpu*/run.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  pkill -KILL -s "$pid" 2>/dev/null || true
done

echo "paused multigpu output=$BASE_OUTPUT_DIR"
