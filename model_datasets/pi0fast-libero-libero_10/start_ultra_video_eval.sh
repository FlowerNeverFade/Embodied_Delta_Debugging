#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
OUTPUT_DIR="${ULTRA_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_targeted_k1_20260527}"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/master.log"
nohup setsid bash -c "cd '$RUN_DIR' && exec ./run_ultra_video_eval.sh" \
  >> "$OUTPUT_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR/run.pid"
echo "started pid=$(cat "$OUTPUT_DIR/run.pid")"
echo "log=$OUTPUT_DIR/master.log"
echo "summary=$OUTPUT_DIR/summary.json"
echo "manifest=$OUTPUT_DIR/manifest.jsonl"
