#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_repair_curriculum_v2_pilot40_multigpu_20260526}"
GPU_IDS=(${MULTIGPU_GPU_IDS:-0 1 2})
PORTS=(${MULTIGPU_POLICY_PORTS:-8040 8041 8042})

for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  shard_output="$BASE_OUTPUT_DIR/gpu${gpu}"
  echo "pausing shard=$i gpu=$gpu port=$port output=$shard_output"
  PILOT_OUTPUT_DIR="$shard_output" POLICY_PORT="$port" "$RUN_DIR/pause_pilot_video_eval.sh" || true
done

echo "paused multigpu output_dir=$BASE_OUTPUT_DIR"
