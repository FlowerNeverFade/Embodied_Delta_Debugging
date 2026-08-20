#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$RUN_DIR/outputs/smoke}" \
TASK_IDS="${SMOKE_TASK_IDS:-0}" \
INIT_STATE_IDS="${SMOKE_INIT_STATE_IDS:-0}" \
SEEDS="${SMOKE_SEEDS:-7}" \
"$RUN_DIR/run_foreground.sh" \
  --max-cases 1 \
  --record-video \
  --require-video \
  --video-dir "${SMOKE_VIDEO_DIR:-${SMOKE_OUTPUT_DIR:-$RUN_DIR/outputs/smoke}/videos}" \
  --video-fps 30 \
  --video-quality 10 \
  --camera-size 512 \
  --event-window 32 \
  --case-timeout-seconds 1800 \
  --scripted-expert-repair-max-steps 180 \
  "$@"
