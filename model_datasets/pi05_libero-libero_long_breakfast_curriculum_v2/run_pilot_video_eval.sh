#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
source "$RUN_DIR/config.env"

OUTPUT_DIR="${PILOT_OUTPUT_DIR:-$RUN_DIR/outputs/risk_repair_curriculum_v2_pilot40_20260526}"
REPORT_DIR="$OUTPUT_DIR/reports"
LOG_DIR="$OUTPUT_DIR/logs"
VIDEO_DIR="$OUTPUT_DIR/videos"

TASK_IDS="${PILOT_TASK_IDS:-0,1,2,3}"
INIT_STATE_IDS="${PILOT_INIT_STATE_IDS:-0,1,2,3,4}"
SEEDS="${PILOT_SEEDS:-7,17}"
MAX_CASES="${PILOT_MAX_CASES:-40}"
POSITIVE_TARGET="${PILOT_POSITIVE_TARGET:-20}"
MIN_CASES_BEFORE_STOP="${PILOT_MIN_CASES_BEFORE_STOP:-40}"
CASE_ORDER_SEED="${PILOT_CASE_ORDER_SEED:-20260526}"
VIDEO_FPS="${PILOT_VIDEO_FPS:-30}"
VIDEO_QUALITY="${PILOT_VIDEO_QUALITY:-10}"
VIDEO_EVERY_N="${PILOT_VIDEO_EVERY_N:-1}"
VIDEO_CODEC="${PILOT_VIDEO_CODEC:-libx264}"
VIDEO_CAMERA="${PILOT_VIDEO_CAMERA:-agentview_image}"
CAMERA_SIZE="${PILOT_CAMERA_SIZE:-512}"
TRAIN_STEPS="${PILOT_TRAIN_STEPS:-500}"
CASE_TIMEOUT_SECONDS="${PILOT_CASE_TIMEOUT_SECONDS:-1800}"
REPAIR_REPLAY_TRIALS="${PILOT_REPAIR_REPLAY_TRIALS:-1}"
EXPERT_REPAIR_MAX_STEPS="${PILOT_EXPERT_REPAIR_MAX_STEPS:-180}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
export CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR" "$VIDEO_DIR"
cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [[ -n "${ACTION_TOKENIZER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--action-tokenizer-path "$ACTION_TOKENIZER_PATH")
fi
if [[ -n "${TEXT_TOKENIZER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--text-tokenizer-path "$TEXT_TOKENIZER_PATH")
fi
if [[ -n "${PYTORCH_DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--pytorch-device "$PYTORCH_DEVICE")
fi
if [[ -n "${PYTORCH_COMPILE_MODE:-}" ]]; then
  EXTRA_ARGS+=(--pytorch-compile-mode "$PYTORCH_COMPILE_MODE")
fi

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /root/autodl-tmp/envs/libero38/bin/python -u "$PROJECT_ROOT/run_risk_critic_large_eval.py" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --log-dir "$LOG_DIR" \
    --policy-port "$POLICY_PORT" \
    --launch-policy-server \
    --policy-server-kind "$POLICY_SERVER_KIND" \
    --policy-config "$POLICY_CONFIG" \
    --policy-dir "$POLICY_DIR" \
    --task-suite-name "$DATASET_NAME" \
    --task-ids "$TASK_IDS" \
    --init-state-ids "$INIT_STATE_IDS" \
    --seeds "$SEEDS" \
    --max-cases "$MAX_CASES" \
    --positive-target "$POSITIVE_TARGET" \
    --min-cases-before-positive-stop "$MIN_CASES_BEFORE_STOP" \
    --shuffle-cases \
    --case-order-seed "$CASE_ORDER_SEED" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --xla-mem-fraction "$XLA_MEM_FRACTION" \
    --search-replay-trials 1 \
    --confirm-replay-trials 5 \
    --repair-replay-trials "$REPAIR_REPLAY_TRIALS" \
    --scripted-expert-repair-max-steps "$EXPERT_REPAIR_MAX_STEPS" \
    --replay-trials 5 \
    --continuation recorded \
    --event-window 32 \
    --case-timeout-seconds "$CASE_TIMEOUT_SECONDS" \
    --train-steps "$TRAIN_STEPS" \
    --record-video \
    --require-video \
    --video-dir "$VIDEO_DIR" \
    --video-camera "$VIDEO_CAMERA" \
    --video-fps "$VIDEO_FPS" \
    --video-every-n "$VIDEO_EVERY_N" \
    --video-codec "$VIDEO_CODEC" \
    --video-quality "$VIDEO_QUALITY" \
    --camera-size "$CAMERA_SIZE" \
    "${EXTRA_ARGS[@]}" \
    "$@"
