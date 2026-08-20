#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="${ROOT:-/data2/yanghaoyun/research/Embodied_Delta_Debugging}"
CODE="$ROOT/v4/code"
PY="${PY:-/data2/yanghaoyun/envs/libero38/bin/python}"
LEROBOT_PY="${LEROBOT_PY:-/data2/yanghaoyun/miniconda3/bin/python}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-/data2/yanghaoyun/research/openpi/packages/openpi-client/src}"
LIBERO_SRC="${LIBERO_SRC:-/data2/yanghaoyun/research/LIBERO}"
OUT="${OUT:-$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/v4_keyframe_repair_hunt20_20260604}"

POLICY_DIR="${POLICY_DIR:-$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay}"
ACTION_TOKENIZER="${ACTION_TOKENIZER:-$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-/data2/yanghaoyun/research/VLA_SKILL/model/google/paligemma-3b-pt-224}"
DEMO_ROOT="${DEMO_ROOT:-/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero}"

GPUS="${GPUS:-0 1 2 3 4}"
PORT_BASE="${PORT_BASE:-8340}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-20}"
SWEEP_TRIALS="${SWEEP_TRIALS:-5}"
KEYFRAMES_PER_CASE="${KEYFRAMES_PER_CASE:-21}"
SWEEP_WINDOW="${SWEEP_WINDOW:-64}"
MAKE_VIDEOS="${MAKE_VIDEOS:-1}"

export EDD_PROJECT_ROOT="$ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero_config"
export OPENPI_PYTHON="$PY"
export LEROBOT_PYTHON="$LEROBOT_PY"
export OPENPI_CLIENT_SRC
export PYTHONPATH="$CODE:$OPENPI_CLIENT_SRC:$LIBERO_SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

if [[ -d /root/autodl-tmp/egl_compat ]]; then
  export LD_LIBRARY_PATH="/root/autodl-tmp/egl_compat:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUT"/{logs,sweep_shards,videos}
echo "$$" > "$OUT/resume_sweep.pid"
: >> "$OUT/resume_sweep.log"

log() {
  echo "$(date -Is) $*" | tee -a "$OUT/resume_sweep.log"
}

port_open() {
  local port="$1"
  "$PY" - "$port" <<'PY'
import socket
import sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
}

wait_port() {
  local port="$1"
  for _ in $(seq 1 450); do
    if port_open "$port"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

require_ports_free() {
  local gpu
  for gpu in $GPUS; do
    local port=$((PORT_BASE + gpu))
    if port_open "$port"; then
      log "ERROR port $port already open"
      exit 3
    fi
  done
}

policy_pids=()
cleanup() {
  local pid
  for pid in "${policy_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

start_policy_server() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/logs/gpu${gpu}"
  mkdir -p "$shard"
  log "starting policy server gpu=$gpu port=$port"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    nohup "$LEROBOT_PY" "$CODE/serve_lerobot_pi0fast_policy.py" \
      --port "$port" \
      --policy-dir "$POLICY_DIR" \
      --device cuda \
      --compile-mode none \
      --action-tokenizer-path "$ACTION_TOKENIZER" \
      --text-tokenizer-path "$TEXT_TOKENIZER" \
      --local-files-only \
      > "$shard/resume_policy_server.log" 2>&1 &
  local pid=$!
  policy_pids+=("$pid")
  echo "$pid" > "$shard/resume_policy_server.pid"
  wait_port "$port"
}

common_args() {
  local gpu="$1"
  local port="$2"
  printf '%s\n' \
    --output-dir "$OUT" \
    --task-suite-name libero_10 \
    --task-ids "0,1,2,3,4,5,6,7,8,9" \
    --init-state-ids "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49" \
    --seeds "7,17,27,37,47,57,67,77,87,97" \
    --policy-port "$port" \
    --policy-dir "$POLICY_DIR" \
    --demo-dataset-root "$DEMO_ROOT" \
    --gpu-id "$gpu" \
    --accepted-target "$ACCEPTED_TARGET" \
    --sweep-trials "$SWEEP_TRIALS" \
    --keyframes-per-case "$KEYFRAMES_PER_CASE" \
    --sweep-window "$SWEEP_WINDOW"
}

run_sweep_worker() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/logs/gpu${gpu}"
  local args_file="$shard/resume_sweep_args.txt"
  local make_video_args=()
  if [[ "$MAKE_VIDEOS" == "1" ]]; then
    make_video_args+=(--make-videos)
  fi
  common_args "$gpu" "$port" > "$args_file"
  log "resume sweep launch gpu=$gpu port=$port shard_index=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$PY" -u "$CODE/keyframe_repair_hunt20.py" sweep-shard @"$args_file" \
      --shard-index "$gpu" \
      --num-shards 5 \
      "${make_video_args[@]}" \
      > "$shard/resume_sweep_worker.log" 2>&1 &
  echo $! > "$shard/resume_sweep_worker.pid"
}

wait_pid_files() {
  local status=0
  local file
  for file in "$@"; do
    local pid
    pid="$(cat "$file")"
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

require_ports_free
for gpu in $GPUS; do
  start_policy_server "$gpu" "$((PORT_BASE + gpu))"
done

pid_files=()
for gpu in $GPUS; do
  run_sweep_worker "$gpu" "$((PORT_BASE + gpu))"
  pid_files+=("$OUT/logs/gpu${gpu}/resume_sweep_worker.pid")
done

if wait_pid_files "${pid_files[@]}"; then
  log "resume sweep workers finished status=0"
else
  log "resume sweep workers finished with nonzero status; finalizing completed rows anyway"
fi

"$PY" "$CODE/keyframe_repair_hunt20.py" finalize-sweep \
  --output-dir "$OUT" \
  --accepted-target "$ACCEPTED_TARGET" \
  --sweep-trials "$SWEEP_TRIALS"

log "resume sweep finalize done output=$OUT"
