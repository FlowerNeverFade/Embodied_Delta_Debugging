#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
CODE_ROOT="$PROJECT_ROOT/prototype/code"
source "$RUN_DIR/config.env"

OUTPUT_DIR="${ULTRA_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_20260525}"
REPORT_DIR="$OUTPUT_DIR/reports"
LOG_DIR="$OUTPUT_DIR/logs"
VIDEO_DIR="$OUTPUT_DIR/videos"

TASK_IDS="${ULTRA_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
INIT_STATE_IDS="${ULTRA_INIT_STATE_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49}"
SEEDS="${ULTRA_SEEDS:-7,17,27,37,47,57,67,77,87,97}"
MAX_CASES="${ULTRA_MAX_CASES:-5000}"
POSITIVE_TARGET="${ULTRA_POSITIVE_TARGET:-200}"
MIN_CASES_BEFORE_STOP="${ULTRA_MIN_CASES_BEFORE_STOP:-900}"
CASE_ORDER_SEED="${ULTRA_CASE_ORDER_SEED:-20260525}"
VIDEO_FPS="${ULTRA_VIDEO_FPS:-30}"
VIDEO_QUALITY="${ULTRA_VIDEO_QUALITY:-10}"
VIDEO_EVERY_N="${ULTRA_VIDEO_EVERY_N:-1}"
VIDEO_CODEC="${ULTRA_VIDEO_CODEC:-libx264}"
VIDEO_CAMERA="${ULTRA_VIDEO_CAMERA:-agentview_image}"
CAMERA_SIZE="${ULTRA_CAMERA_SIZE:-512}"
TRAIN_STEPS="${ULTRA_TRAIN_STEPS:-500}"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR" "$VIDEO_DIR"
cd "$PROJECT_ROOT"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /root/autodl-tmp/envs/libero38/bin/python -u "$CODE_ROOT/run_risk_critic_large_eval.py" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --log-dir "$LOG_DIR" \
    --policy-port "$POLICY_PORT" \
    --launch-policy-server \
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
    --replay-trials 5 \
    --continuation recorded \
    --event-window 32 \
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
    "$@"
