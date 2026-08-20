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
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
SEEDS="${SEEDS:-7,17,27,37,47,57,67,77,87,97}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-20}"
SEARCH_REPLAY_TRIALS="${SEARCH_REPLAY_TRIALS:-1}"
SEARCH_CONFIRM_TRIALS="${SEARCH_CONFIRM_TRIALS:-1}"
CANDIDATE_SCREEN_TRIALS="${CANDIDATE_SCREEN_TRIALS:-1}"
ACCEPT_TRIALS="${ACCEPT_TRIALS:-5}"
SWEEP_TRIALS="${SWEEP_TRIALS:-5}"
MAX_KEYFRAME_CANDIDATES="${MAX_KEYFRAME_CANDIDATES:-32}"
KEYFRAMES_PER_CASE="${KEYFRAMES_PER_CASE:-21}"
SWEEP_WINDOW="${SWEEP_WINDOW:-64}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
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

mkdir -p "$OUT"/{logs,sweep_shards,videos,case_results,rollout_archives}
ln -sfn "$OUT" "$ROOT/v4/outputs/keyframe_repair_hunt20_latest"
echo "$$" > "$OUT/run.pid"
: > "$OUT/master.log"

log() {
  echo "$(date -Is) $*" | tee -a "$OUT/master.log"
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
      log "ERROR port $port already open; refusing to reuse an unverified policy server"
      exit 3
    fi
  done
}

init_ids_for_gpu() {
  case "$1" in
    0) echo "0,1,2,3,4,5,6,7,8,9" ;;
    1) echo "10,11,12,13,14,15,16,17,18,19" ;;
    2) echo "20,21,22,23,24,25,26,27,28,29" ;;
    3) echo "30,31,32,33,34,35,36,37,38,39" ;;
    4) echo "40,41,42,43,44,45,46,47,48,49" ;;
    *) echo "" ;;
  esac
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
      > "$shard/policy_server.log" 2>&1 &
  local pid=$!
  policy_pids+=("$pid")
  echo "$pid" > "$shard/policy_server.pid"
  wait_port "$port"
}

common_probe_args() {
  local gpu="$1"
  local port="$2"
  local init_ids="$3"
  local force_args=()
  if [[ "$FORCE" == "1" ]]; then
    force_args+=(--force)
  fi
  printf '%s\n' \
    --output-dir "$OUT" \
    --task-suite-name libero_10 \
    --task-ids "$TASK_IDS" \
    --init-state-ids "$init_ids" \
    --seeds "$SEEDS" \
    --policy-port "$port" \
    --policy-dir "$POLICY_DIR" \
    --demo-dataset-root "$DEMO_ROOT" \
    --gpu-id "$gpu" \
    --accepted-target "$ACCEPTED_TARGET" \
    --search-replay-trials "$SEARCH_REPLAY_TRIALS" \
    --search-confirm-trials "$SEARCH_CONFIRM_TRIALS" \
    --candidate-screen-trials "$CANDIDATE_SCREEN_TRIALS" \
    --accept-trials "$ACCEPT_TRIALS" \
    --sweep-trials "$SWEEP_TRIALS" \
    --max-keyframe-candidates "$MAX_KEYFRAME_CANDIDATES" \
    --keyframes-per-case "$KEYFRAMES_PER_CASE" \
    --sweep-window "$SWEEP_WINDOW" \
    "${force_args[@]}"
}

run_hunt_worker() {
  local gpu="$1"
  local port="$2"
  local init_ids="$3"
  local shard="$OUT/logs/gpu${gpu}"
  local args_file="$shard/hunt_args.txt"
  mkdir -p "$shard"
  common_probe_args "$gpu" "$port" "$init_ids" > "$args_file"
  log "hunt launch gpu=$gpu port=$port init_ids=$init_ids"
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$PY" -u "$CODE/keyframe_repair_hunt20.py" hunt-worker @"$args_file" \
      > "$shard/hunt_worker.log" 2>&1 &
  echo $! > "$shard/hunt_worker.pid"
}

run_sweep_worker() {
  local gpu="$1"
  local port="$2"
  local init_ids="$3"
  local shard="$OUT/logs/gpu${gpu}"
  local args_file="$shard/sweep_args.txt"
  local make_video_args=()
  if [[ "$MAKE_VIDEOS" == "1" ]]; then
    make_video_args+=(--make-videos)
  fi
  common_probe_args "$gpu" "$port" "$init_ids" > "$args_file"
  log "sweep launch gpu=$gpu port=$port shard_index=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$PY" -u "$CODE/keyframe_repair_hunt20.py" sweep-shard @"$args_file" \
      --shard-index "$gpu" \
      --num-shards 5 \
      "${make_video_args[@]}" \
      > "$shard/sweep_worker.log" 2>&1 &
  echo $! > "$shard/sweep_worker.pid"
}

wait_pid_files() {
  local label="$1"
  shift
  local status=0
  local file
  for file in "$@"; do
    local pid
    pid="$(cat "$file")"
    if ! wait "$pid"; then
      status=1
    fi
  done
  log "$label finished status=$status"
  return "$status"
}

if [[ "$DRY_RUN" == "1" ]]; then
  "$PY" "$CODE/keyframe_repair_hunt20.py" dry-run \
    --output-dir "$OUT" \
    --task-suite-name libero_10 \
    --task-ids "$TASK_IDS" \
    --init-state-ids "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49" \
    --seeds "$SEEDS" \
    --accepted-target "$ACCEPTED_TARGET" \
    --sweep-trials "$SWEEP_TRIALS" \
    --keyframes-per-case "$KEYFRAMES_PER_CASE"
  exit 0
fi

require_ports_free

for gpu in $GPUS; do
  start_policy_server "$gpu" "$((PORT_BASE + gpu))"
done

hunt_pid_files=()
for gpu in $GPUS; do
  init_ids="$(init_ids_for_gpu "$gpu")"
  run_hunt_worker "$gpu" "$((PORT_BASE + gpu))" "$init_ids"
  hunt_pid_files+=("$OUT/logs/gpu${gpu}/hunt_worker.pid")
done
wait_pid_files hunt "${hunt_pid_files[@]}"

"$PY" "$CODE/keyframe_repair_hunt20.py" finalize-accepted \
  --output-dir "$OUT" \
  --accepted-target "$ACCEPTED_TARGET"

accepted_count="$("$PY" - "$OUT/accepted_cases_manifest.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
try:
    print(len(json.loads(path.read_text(encoding="utf-8"))))
except Exception:
    print(0)
PY
)"
log "accepted_count=$accepted_count target=$ACCEPTED_TARGET"
if [[ "$accepted_count" -le 0 ]]; then
  log "ERROR no accepted cases; sweep skipped"
  exit 5
fi

sweep_pid_files=()
for gpu in $GPUS; do
  init_ids="$(init_ids_for_gpu "$gpu")"
  run_sweep_worker "$gpu" "$((PORT_BASE + gpu))" "$init_ids"
  sweep_pid_files+=("$OUT/logs/gpu${gpu}/sweep_worker.pid")
done
wait_pid_files sweep "${sweep_pid_files[@]}"

"$PY" "$CODE/keyframe_repair_hunt20.py" finalize-sweep \
  --output-dir "$OUT" \
  --accepted-target "$ACCEPTED_TARGET" \
  --sweep-trials "$SWEEP_TRIALS"

log "done output=$OUT"
