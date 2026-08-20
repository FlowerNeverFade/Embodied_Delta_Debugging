#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_repair_curriculum_v2_pi05_pilot40_multigpu_20260526}"
mkdir -p "$BASE_OUTPUT_DIR"

GPU_IDS=(${MULTIGPU_GPU_IDS:-0 1 2})
PORTS=(${MULTIGPU_POLICY_PORTS:-8050 8051 8052})
INIT_SHARDS=(
  "${MULTIGPU_INIT_SHARD_0:-0,1}"
  "${MULTIGPU_INIT_SHARD_1:-2,3}"
  "${MULTIGPU_INIT_SHARD_2:-4}"
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
  PILOT_OUTPUT_DIR="$shard_output" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  POLICY_PORT="$port" \
  PILOT_INIT_STATE_IDS="${INIT_SHARDS[$i]}" \
  "$RUN_DIR/start_pilot_video_eval.sh"
done

cat > "$BASE_OUTPUT_DIR/multigpu_shards.json" <<JSON
{
  "schema_version": "shed-cfs-curriculum-v2-multigpu-shards-v1",
  "base_output_dir": "$BASE_OUTPUT_DIR",
  "note": "Each shard has its own policy server, manifest, reports, videos, and postprocess outputs."
}
JSON

echo "multigpu_output=$BASE_OUTPUT_DIR"
