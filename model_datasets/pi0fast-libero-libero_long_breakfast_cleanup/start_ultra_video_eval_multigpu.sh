#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_long_breakfast_video_causal_v2_multigpu_20260526}"
mkdir -p "$BASE_OUTPUT_DIR"

GPU_IDS=(${MULTIGPU_GPU_IDS:-0 1 2})
PORTS=(${MULTIGPU_POLICY_PORTS:-8030 8031 8032})
INIT_SHARDS=(
  "${MULTIGPU_INIT_SHARD_0:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}"
  "${MULTIGPU_INIT_SHARD_1:-17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33}"
  "${MULTIGPU_INIT_SHARD_2:-34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49}"
)

if [[ "${#GPU_IDS[@]}" -ne "${#PORTS[@]}" ]]; then
  echo "MULTIGPU_GPU_IDS and MULTIGPU_POLICY_PORTS must have the same length" >&2
  exit 2
fi

for i in "${!GPU_IDS[@]}"; do
  if [[ "$i" -ge "${#INIT_SHARDS[@]}" ]]; then
    echo "No init shard configured for shard index $i" >&2
    exit 2
  fi
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  shard_output="$BASE_OUTPUT_DIR/gpu${gpu}"
  mkdir -p "$shard_output"
  echo "starting shard=$i gpu=$gpu port=$port output=$shard_output init_ids=${INIT_SHARDS[$i]}"
  ULTRA_OUTPUT_DIR="$shard_output" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  POLICY_PORT="$port" \
  ULTRA_INIT_STATE_IDS="${INIT_SHARDS[$i]}" \
  "$RUN_DIR/start_ultra_video_eval.sh"
done

cat > "$BASE_OUTPUT_DIR/multigpu_shards.json" <<JSON
{
  "schema_version": "shed-cfs-multigpu-shards-v1",
  "base_output_dir": "$BASE_OUTPUT_DIR",
  "note": "Each shard has its own policy server, manifest, reports, videos, and postprocess outputs for the custom long breakfast task."
}
JSON

echo "multigpu_output=$BASE_OUTPUT_DIR"
