#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
source "$RUN_DIR/config.env"

CONFIRM_OUTPUT_DIR="${CONFIRM_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_confirm_k5_20260527}"
CONFIRM_GPU_IDS=(${CONFIRM_GPU_IDS:-0 1 2})
CONFIRM_POLICY_PORTS=(${CONFIRM_POLICY_PORTS:-8030 8031 8032})
CONFIRM_CASE_TIMEOUT_SECONDS="${CONFIRM_CASE_TIMEOUT_SECONDS:-0}"
CONFIRM_REPAIR_REPLAY_TRIALS="${CONFIRM_REPAIR_REPLAY_TRIALS:-5}"
CONFIRM_CAUSAL_MAX_UNITS="${CONFIRM_CAUSAL_MAX_UNITS:-18}"
CONFIRM_CAUSAL_CONTEXT_BEFORE="${CONFIRM_CAUSAL_CONTEXT_BEFORE:-48}"
CONFIRM_CAUSAL_CONTEXT_AFTER="${CONFIRM_CAUSAL_CONTEXT_AFTER:-8}"
CONFIRM_CAMERA_SIZE="${CONFIRM_CAMERA_SIZE:-512}"
CONFIRM_VIDEO_FPS="${CONFIRM_VIDEO_FPS:-30}"
CONFIRM_VIDEO_QUALITY="${CONFIRM_VIDEO_QUALITY:-10}"
CONFIRM_VIDEO_CODEC="${CONFIRM_VIDEO_CODEC:-libx264}"
CONFIRM_VIDEO_CAMERA="${CONFIRM_VIDEO_CAMERA:-agentview_image}"
CONFIRM_XLA_MEM_FRACTION="${CONFIRM_XLA_MEM_FRACTION:-0.55}"

# Format: task_id,init_state_id,seed. These are frozen from the K=1 v3 targeted run.
CONFIRM_CANDIDATES="${CONFIRM_CANDIDATES:-9,1,27;1,15,7;8,18,7;8,21,17;8,46,17}"

mkdir -p "$CONFIRM_OUTPUT_DIR"
echo "$$" > "$CONFIRM_OUTPUT_DIR/run.pid"

python3 - "$CONFIRM_OUTPUT_DIR" "$CONFIRM_CANDIDATES" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
cases = []
for raw in sys.argv[2].split(";"):
    raw = raw.strip()
    if not raw:
        continue
    task, init, seed = (int(x) for x in raw.split(","))
    cases.append({"task_id": task, "init_state_id": init, "seed": seed})

(out / "candidate_index.json").write_text(
    json.dumps(
        {
            "schema_version": "shed-cfs-causal-v3-confirm-candidate-index-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "risk_critic_ultra_video_causal_v3_targeted_k1_20260527",
            "confirm_replay_trials": 5,
            "repair_replay_trials": 5,
            "cases": cases,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

wait_for_port() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            sys.exit(0)
    except OSError:
        time.sleep(2)
print(f"port {port} did not become ready", file=sys.stderr)
sys.exit(1)
PY
}

write_manifest_row() {
  local manifest="$1"
  local shard="$2"
  local task_id="$3"
  local init_state_id="$4"
  local seed="$5"
  local return_code="$6"
  local elapsed="$7"
  local report_path="$8"
  local log_path="$9"
  python3 - "$manifest" "$shard" "$task_id" "$init_state_id" "$seed" "$return_code" "$elapsed" "$report_path" "$log_path" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest, shard, task_id, init_state_id, seed, return_code, elapsed, report_path, log_path = sys.argv[1:]
task_id = int(task_id)
init_state_id = int(init_state_id)
seed = int(seed)
return_code = int(return_code)
elapsed = float(elapsed)
report = None
path = Path(report_path)
if path.exists():
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        report = None

def b(*keys):
    cur = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, bool) else None

policy_pass = None
demo_pass = None
full_success = None
same = None
positive = 0
status = "probe_failed"
schema = None
video_paths = []
if isinstance(report, dict):
    schema = report.get("schema_version")
    same = b("reproduction_statistics", "same_failure")
    policy_pass = bool(
        report.get("policy_strong_repair_valid_pass")
        or (report.get("causal_validation") or {}).get("policy_strong_repair_valid_pass")
    )
    demo_pass = bool(
        report.get("demo_existence_repair_pass")
        or (report.get("causal_validation") or {}).get("demo_existence_repair_pass")
    )
    full_success = bool(
        report.get("full_success_policy_repair_pass")
        or report.get("full_success_repair_pass")
        or (report.get("causal_validation") or {}).get("full_success_policy_repair_pass")
    )
    positive = sum(
        1 for w in report.get("risk_training_windows") or []
        if int(w.get("label") or 0) == 1
    )
    for rollout in report.get("rollout_summaries") or []:
        if rollout.get("video_path"):
            video_paths.append(rollout["video_path"])
    if policy_pass:
        status = "k5_policy_strong_pass"
    elif same:
        status = "k5_same_failure_only"
    elif return_code in (0, 1):
        status = "k5_nonpass"

row = {
    "schema_version": "shed-cfs-causal-v3-confirm-manifest-v1",
    "shard": shard,
    "task_id": task_id,
    "init_state_id": init_state_id,
    "seed": seed,
    "status": status,
    "return_code": return_code,
    "report_path": str(path) if path.exists() else None,
    "log_path": log_path,
    "report_schema_version": schema,
    "same_failure_pass": same,
    "policy_strong_repair_valid_pass": policy_pass,
    "demo_existence_repair_pass": demo_pass,
    "full_success_repair_pass": full_success,
    "positive_windows": positive,
    "video_paths": video_paths,
    "video_exists": bool(video_paths) and all(Path(p).exists() for p in video_paths),
    "wall_seconds": elapsed,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
with Path(manifest).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
}

worker() {
  local worker_idx="$1"
  local gpu="${CONFIRM_GPU_IDS[$worker_idx]}"
  local port="${CONFIRM_POLICY_PORTS[$worker_idx]}"
  local shard_dir="$CONFIRM_OUTPUT_DIR/gpu${gpu}"
  local report_dir="$shard_dir/reports"
  local log_dir="$shard_dir/logs"
  local video_dir="$shard_dir/videos"
  local manifest="$shard_dir/manifest.jsonl"
  local server_log="$shard_dir/policy_server.log"
  local worker_log="$shard_dir/worker.log"
  mkdir -p "$report_dir" "$log_dir" "$video_dir"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export MUJOCO_EGL_DEVICE_ID="$gpu"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$CONFIRM_XLA_MEM_FRACTION"

  echo "starting policy server gpu=$gpu port=$port" | tee -a "$worker_log"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
      -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /root/miniconda3/bin/python "$PROJECT_ROOT/serve_lerobot_pi0fast_policy.py" \
        --port "$port" \
        --policy-dir "$POLICY_DIR" \
        --device "$PYTORCH_DEVICE" \
        --compile-mode "$PYTORCH_COMPILE_MODE" \
        --action-tokenizer-path "$ACTION_TOKENIZER_PATH" \
        --text-tokenizer-path "$TEXT_TOKENIZER_PATH" \
        --local-files-only \
        > "$server_log" 2>&1 &
  local server_pid="$!"
  echo "$server_pid" > "$shard_dir/policy_server.pid"
  trap 'kill "$server_pid" 2>/dev/null || true' EXIT
  wait_for_port "$port"

  IFS=';' read -r -a cases <<< "$CONFIRM_CANDIDATES"
  for idx in "${!cases[@]}"; do
    if (( idx % ${#CONFIRM_GPU_IDS[@]} != worker_idx )); then
      continue
    fi
    IFS=',' read -r task_id init_state_id seed <<< "${cases[$idx]}"
    local report_path="$report_dir/task$(printf "%02d" "$task_id")_init$(printf "%02d" "$init_state_id")_seed$(printf "%02d" "$seed")_causal_v3_k5.json"
    local log_path="$log_dir/task$(printf "%02d" "$task_id")_init$(printf "%02d" "$init_state_id")_seed$(printf "%02d" "$seed").log"
    local prefix="task$(printf "%02d" "$task_id")_init$(printf "%02d" "$init_state_id")_seed$(printf "%02d" "$seed")_k5"
    local start_ts
    start_ts="$(date +%s)"
    echo "running candidate task=$task_id init=$init_state_id seed=$seed gpu=$gpu" | tee -a "$worker_log"
    set +e
    local timeout_cmd=()
    if [[ "$CONFIRM_CASE_TIMEOUT_SECONDS" != "0" && "$CONFIRM_CASE_TIMEOUT_SECONDS" != "0.0" ]]; then
      timeout_cmd=(timeout "$CONFIRM_CASE_TIMEOUT_SECONDS")
    fi
    "${timeout_cmd[@]}" env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
          -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
          CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        /root/autodl-tmp/envs/libero38/bin/python "$PROJECT_ROOT/pi05_natural_failure_probe.py" \
          --policy-host 127.0.0.1 \
          --policy-port "$port" \
          --policy-config "$POLICY_CONFIG" \
          --policy-checkpoint "$POLICY_DIR" \
          --task-suite-name "$DATASET_NAME" \
          --task-ids "$task_id" \
          --init-state-ids "$init_state_id" \
          --seed "$seed" \
          --replay-trials 5 \
          --search-replay-trials 1 \
          --confirm-replay-trials 5 \
          --repair-replay-trials "$CONFIRM_REPAIR_REPLAY_TRIALS" \
          --event-window 32 \
          --causal-context-before "$CONFIRM_CAUSAL_CONTEXT_BEFORE" \
          --causal-context-after "$CONFIRM_CAUSAL_CONTEXT_AFTER" \
          --causal-max-units "$CONFIRM_CAUSAL_MAX_UNITS" \
          --causal-ablation-trials 5 \
          --continuation recorded \
          --scripted-expert-repair-max-steps 180 \
          --skip-source-repair-if-policy-pass \
          --stop-after-first-repair-valid-core \
          --initial-state-max-attempts 8 \
          --camera-size "$CONFIRM_CAMERA_SIZE" \
          --gpu-id "$gpu" \
          --output "$report_path" \
          --enable-visual-policy-mask \
          --record-video \
          --video-dir "$video_dir" \
          --video-prefix "$prefix" \
          --video-camera "$CONFIRM_VIDEO_CAMERA" \
          --video-fps "$CONFIRM_VIDEO_FPS" \
          --video-every-n 1 \
          --video-codec "$CONFIRM_VIDEO_CODEC" \
          --video-quality "$CONFIRM_VIDEO_QUALITY" \
          > "$log_path" 2>&1
    local rc="$?"
    set -e
    local end_ts elapsed
    end_ts="$(date +%s)"
    elapsed="$((end_ts - start_ts))"
    if [[ "$rc" == "124" ]]; then
      echo "timeout candidate task=$task_id init=$init_state_id seed=$seed" | tee -a "$worker_log"
    fi
    write_manifest_row "$manifest" "gpu${gpu}" "$task_id" "$init_state_id" "$seed" "$rc" "$elapsed" "$report_path" "$log_path" | tee -a "$worker_log"
  done

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

for i in "${!CONFIRM_GPU_IDS[@]}"; do
  worker "$i" > "$CONFIRM_OUTPUT_DIR/gpu${CONFIRM_GPU_IDS[$i]}.outer.log" 2>&1 &
  echo "$!" > "$CONFIRM_OUTPUT_DIR/gpu${CONFIRM_GPU_IDS[$i]}.worker.pid"
done

wait

python3 - "$CONFIRM_OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

out = Path(__import__("sys").argv[1])
rows = []
for mf in sorted(out.glob("gpu*/manifest.jsonl")):
    for line in mf.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))

summary = {
    "schema_version": "shed-cfs-causal-v3-confirm-summary-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(out),
    "processed_cases": len(rows),
    "status_counts": dict(Counter(row.get("status") for row in rows)),
    "same_failure_passes": sum(row.get("same_failure_pass") is True for row in rows),
    "policy_strong_repair_valid_passes": sum(row.get("policy_strong_repair_valid_pass") is True for row in rows),
    "full_success_repair_passes": sum(row.get("full_success_repair_pass") is True for row in rows),
    "positive_windows": sum(int(row.get("positive_windows") or 0) for row in rows),
    "rows": rows,
}
(out / "confirm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
