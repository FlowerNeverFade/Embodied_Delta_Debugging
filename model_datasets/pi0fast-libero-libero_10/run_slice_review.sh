#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
source "$RUN_DIR/config.env"

REPORT_DIR="${REVIEW_REPORT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v2_20260525/reports}"
OUTPUT_DIR="${REVIEW_OUTPUT_DIR:-$RUN_DIR/outputs/slice_review_$(date +%Y%m%d_%H%M%S)}"
REPLAY_TRIALS="${REVIEW_REPLAY_TRIALS:-5}"
MAX_CASES="${REVIEW_MAX_CASES:-0}"
CONTEXT_SECONDS="${REVIEW_CONTEXT_SECONDS:-3}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
export CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /root/autodl-tmp/envs/libero38/bin/python -u "$PROJECT_ROOT/slice_review_export.py" \
    --report-dir "$REPORT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --task-suite-name "$DATASET_NAME" \
    --policy-port "$POLICY_PORT" \
    --policy-config "$POLICY_CONFIG" \
    --policy-dir "$POLICY_DIR" \
    --action-tokenizer-path "$ACTION_TOKENIZER_PATH" \
    --text-tokenizer-path "$TEXT_TOKENIZER_PATH" \
    --launch-policy-server \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --pytorch-device "$PYTORCH_DEVICE" \
    --pytorch-compile-mode "$PYTORCH_COMPILE_MODE" \
    --camera-size 512 \
    --video-fps 30 \
    --video-quality 10 \
    --context-seconds "$CONTEXT_SECONDS" \
    --replay-trials-to-record "$REPLAY_TRIALS" \
    --max-cases "$MAX_CASES" \
    "$@"
