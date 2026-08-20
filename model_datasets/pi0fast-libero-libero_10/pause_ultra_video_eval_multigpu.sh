#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_targeted_k1_20260527}"
GPU_IDS=(${MULTIGPU_GPU_IDS:-0 1 2})
PORTS=(${MULTIGPU_POLICY_PORTS:-8020 8021 8022})

for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  shard_output="$BASE_OUTPUT_DIR/gpu${gpu}"
  echo "pausing shard=$i gpu=$gpu port=$port output=$shard_output"
  ULTRA_OUTPUT_DIR="$shard_output" POLICY_PORT="$port" "$RUN_DIR/pause_ultra_video_eval.sh" || true
done

echo "paused multigpu output_dir=$BASE_OUTPUT_DIR"
