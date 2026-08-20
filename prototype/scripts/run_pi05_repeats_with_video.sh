#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="$PROJECT_ROOT/prototype/code"
OPENPI_ROOT="/root/autodl-tmp/research/openpi"
OPENPI_PYTHON="/root/autodl-tmp/research/openpi/.venv/bin/python"
LIBERO_PYTHON="/root/autodl-tmp/envs/libero38/bin/python"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/pi05_repeats_video}"
VIDEO_DIR="${VIDEO_DIR:-$OUTPUT_DIR/videos}"
LOG_DIR="$OUTPUT_DIR/logs"
POLICY_PORT="${POLICY_PORT:-8000}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
TASK_IDS="${TASK_IDS:-8}"
INIT_STATE_IDS="${INIT_STATE_IDS:-0}"
SEED="${SEED:-7}"
REPEATS="${REPEATS:-10}"
START_REPEAT="${START_REPEAT:-1}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_CAMERA="${VIDEO_CAMERA:-agentview_image}"

mkdir -p "$OUTPUT_DIR" "$VIDEO_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

port_open() {
  "$LIBERO_PYTHON" - "$POLICY_HOST" "$POLICY_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

wait_for_port() {
  for _ in $(seq 1 180); do
    if port_open; then
      return 0
    fi
    sleep 2
  done
  echo "Policy server did not become ready on $POLICY_HOST:$POLICY_PORT" >&2
  return 1
}

POLICY_PID=""
if port_open; then
  echo "Using existing policy server at $POLICY_HOST:$POLICY_PORT"
else
  echo "Starting policy server at $POLICY_HOST:$POLICY_PORT"
  (
    cd "$OPENPI_ROOT"
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.55}" \
      "$OPENPI_PYTHON" scripts/serve_policy.py \
        --port "$POLICY_PORT" \
        policy:checkpoint \
        --policy.config="$POLICY_CONFIG" \
        --policy.dir="$POLICY_DIR"
  ) > "$OUTPUT_DIR/policy_server.log" 2>&1 &
  POLICY_PID="$!"
  echo "$POLICY_PID" > "$OUTPUT_DIR/policy_server.pid"
  wait_for_port
fi

cleanup() {
  if [[ -n "$POLICY_PID" ]]; then
    kill "$POLICY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

END_REPEAT=$((START_REPEAT + REPEATS - 1))
for repeat_id in $(seq "$START_REPEAT" "$END_REPEAT"); do
  repeat_tag="$(printf '%02d' "$repeat_id")"
  output_path="$OUTPUT_DIR/task8_init0_repeat_${repeat_tag}_causal_v1.json"
  log_path="$LOG_DIR/task8_init0_repeat_${repeat_tag}.log"
  video_prefix="task8_init0_repeat_${repeat_tag}"
  echo "RUN $repeat_tag start $(date --iso-8601=seconds)" | tee "$log_path"
  if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    "$LIBERO_PYTHON" "$CODE_ROOT/pi05_natural_failure_probe.py" \
      --policy-host "$POLICY_HOST" \
      --policy-port "$POLICY_PORT" \
      --policy-config "$POLICY_CONFIG" \
      --policy-checkpoint "$POLICY_DIR" \
      --task-suite-name "$TASK_SUITE_NAME" \
      --task-ids "$TASK_IDS" \
      --init-state-ids "$INIT_STATE_IDS" \
      --seed "$SEED" \
      --event-window 32 \
      --continuation recorded \
      --record-video \
      --video-dir "$VIDEO_DIR" \
      --video-prefix "$video_prefix" \
      --video-fps "$VIDEO_FPS" \
      --video-camera "$VIDEO_CAMERA" \
      --output "$output_path" \
      >> "$log_path" 2>&1; then
    echo "RUN $repeat_tag status=pass" | tee -a "$log_path"
  else
    status="$?"
    echo "RUN $repeat_tag status=nonpass exit=$status" | tee -a "$log_path"
  fi
done

find "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' | sort > "$OUTPUT_DIR/videos.txt"
echo "videos=$(wc -l < "$OUTPUT_DIR/videos.txt")"
echo "output_dir=$OUTPUT_DIR"
