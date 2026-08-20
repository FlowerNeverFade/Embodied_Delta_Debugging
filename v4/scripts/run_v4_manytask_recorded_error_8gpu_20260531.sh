#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="${ROOT:-/data2/yanghaoyun/research/Embodied_Delta_Debugging}"
CODE="$ROOT/v4/code"
PY="${PY:-/data2/yanghaoyun/envs/libero38/bin/python}"
LEROBOT_PY="${LEROBOT_PY:-/data2/yanghaoyun/miniconda3/bin/python}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-/data2/yanghaoyun/research/openpi/packages/openpi-client/src}"
LIBERO_SRC="${LIBERO_SRC:-/data2/yanghaoyun/research/LIBERO}"
OUT="${OUT:-$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/v4_manytask_recorded_error_8gpu_20260531}"

POLICY_DIR="${POLICY_DIR:-$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay}"
ACTION_TOKENIZER="${ACTION_TOKENIZER:-$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-/data2/yanghaoyun/research/VLA_SKILL/model/google/paligemma-3b-pt-224}"
DEMO_ROOT="${DEMO_ROOT:-/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
SEEDS="${SEEDS:-7,17,27}"
MAX_CONFIRM_CANDIDATES="${MAX_CONFIRM_CANDIDATES:-160}"
MAX_CONFIRM_PER_TASK="${MAX_CONFIRM_PER_TASK:-24}"
SEARCH_CASE_TIMEOUT_SECONDS="${SEARCH_CASE_TIMEOUT_SECONDS:-1800}"
CONFIRM_CASE_TIMEOUT_SECONDS="${CONFIRM_CASE_TIMEOUT_SECONDS:-0}"
SEARCH_PORT_BASE="${SEARCH_PORT_BASE:-8200}"
CONFIRM_PORT_BASE="${CONFIRM_PORT_BASE:-8210}"
REVIEW_PORT_BASE="${REVIEW_PORT_BASE:-8220}"

export EDD_PROJECT_ROOT="$ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero_config"
export OPENPI_PYTHON="$PY"
export LEROBOT_PYTHON="$LEROBOT_PY"
export OPENPI_CLIENT_SRC
export PYTHONPATH="$CODE:$OPENPI_CLIENT_SRC:$LIBERO_SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"/{search,confirm,review,logs}
ln -sfn "$OUT" "$ROOT/v4/outputs/manytask_recorded_error_8gpu_latest"
echo "$$" > "$OUT/run.pid"

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

require_ports_free() {
  local base="$1"
  local gpu
  for gpu in $GPUS; do
    local port=$((base + gpu))
    if port_open "$port"; then
      log "ERROR port $port is already open; refusing to risk reusing a wrong policy server"
      exit 3
    fi
  done
}

init_ids_for_gpu() {
  case "$1" in
    0) echo "0,1,2,3,4,5,6" ;;
    1) echo "7,8,9,10,11,12,13" ;;
    2) echo "14,15,16,17,18,19" ;;
    3) echo "20,21,22,23,24,25" ;;
    4) echo "26,27,28,29,30,31" ;;
    5) echo "32,33,34,35,36,37" ;;
    6) echo "38,39,40,41,42,43" ;;
    7) echo "44,45,46,47,48,49" ;;
    *) echo "" ;;
  esac
}

start_search_shard() {
  local gpu="$1"
  local init_ids
  init_ids="$(init_ids_for_gpu "$gpu")"
  if [[ -z "$init_ids" ]]; then
    log "ERROR no init shard mapping for gpu=$gpu"
    exit 4
  fi
  local port=$((SEARCH_PORT_BASE + gpu))
  local shard="$OUT/search/gpu${gpu}"
  mkdir -p "$shard"/{reports,logs,videos}
  log "search launch gpu=$gpu port=$port init_ids=$init_ids"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
      "$PY" -u "$CODE/run_risk_critic_large_eval.py" \
        --output-dir "$shard" \
        --report-dir "$shard/reports" \
        --log-dir "$shard/logs" \
        --manifest-path "$shard/manifest.jsonl" \
        --cost-summary-path "$shard/cost_summary.json" \
        --export-path "$shard/risk_critic_full_v1.jsonl" \
        --train-output "$shard/risk_critic_full_metrics.json" \
        --summary-path "$shard/summary.json" \
        --policy-port "$port" \
        --launch-policy-server \
        --policy-server-kind lerobot_pi0fast \
        --policy-config pi0_fast_libero \
        --policy-dir "$POLICY_DIR" \
        --action-tokenizer-path "$ACTION_TOKENIZER" \
        --text-tokenizer-path "$TEXT_TOKENIZER" \
        --pytorch-device cuda \
        --pytorch-compile-mode none \
        --task-suite-name libero_10 \
        --task-ids "$TASK_IDS" \
        --init-state-ids "$init_ids" \
        --seeds "$SEEDS" \
        --max-cases 100000 \
        --positive-target 0 \
        --min-cases-before-positive-stop 0 \
        --shuffle-cases \
        --case-order-seed 20260531 \
        --cuda-visible-devices "$gpu" \
        --xla-mem-fraction 0.55 \
        --search-replay-trials 1 \
        --confirm-replay-trials 1 \
        --repair-replay-trials 1 \
        --replay-trials 1 \
        --continuation recorded \
        --event-window 32 \
        --causal-context-before 48 \
        --causal-context-after 8 \
        --causal-max-units 18 \
        --causal-ablation-trials 1 \
        --causal-ablation-strategies hold \
        --repair-scheduling-mode pass_hunt \
        --demo-repair-timeout-seconds 30 \
        --case-timeout-seconds "$SEARCH_CASE_TIMEOUT_SECONDS" \
        --train-steps 1 \
        --postprocess-scope current-rows \
        --resume \
        --record-video \
        --require-video \
        --video-dir "$shard/videos" \
        --video-camera agentview_image \
        --video-fps 30 \
        --video-every-n 1 \
        --video-codec libx264 \
        --video-quality 10 \
        --camera-size 512 \
        --skip-source-repair-if-policy-pass \
        --stop-after-first-repair-valid-core \
        --defer-source-repair \
        --enable-visual-policy-mask
  ) > "$shard/master.log" 2>&1 &
  echo $! > "$shard/run.pid"
}

wait_shards() {
  local phase="$1"
  shift
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  log "$phase finished status=$status"
  return "$status"
}

select_confirm_candidates() {
  log "selecting K=5 confirmation candidates"
  "$PY" - "$OUT" "$MAX_CONFIRM_CANDIDATES" "$MAX_CONFIRM_PER_TASK" <<'PY'
import collections
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
max_total = int(sys.argv[2])
max_per_task = int(sys.argv[3])
rows = []
for path in sorted((out / "search").glob("gpu*/reports/*_causal_v4.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    selected = report.get("selected_failed_rollout") or {}
    if not selected:
        continue
    task = selected.get("task_id")
    init = selected.get("init_state_id")
    seed = selected.get("reset_seed")
    if task is None or init is None or seed is None:
        continue
    natural_failure = selected.get("success") is False
    if not natural_failure:
        continue
    causal = report.get("causal_validation") or {}
    repro = report.get("reproduction_statistics") or {}
    same_repro = bool(repro.get("same_failure"))
    same_nec = bool(report.get("same_failure_necessity_pass") or causal.get("same_failure_necessity_pass"))
    policy = bool(report.get("policy_strong_repair_valid_pass") or causal.get("policy_strong_repair_valid_pass"))
    full = bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass"))
    if full or policy:
        priority = 0
    elif same_nec:
        priority = 1
    elif same_repro:
        priority = 2
    else:
        priority = 3
    rows.append({
        "task_id": int(task),
        "init_state_id": int(init),
        "seed": int(seed),
        "source_report": str(path),
        "source_priority": priority,
        "source_same_repro": same_repro,
        "source_same_failure_necessity_pass": same_nec,
        "source_policy_strong_repair_valid_pass": policy,
        "source_full_success_repair_pass": full,
        "failure_type": (selected.get("failure_signature") or {}).get("failure_type")
            or (report.get("failure_signature") or {}).get("failure_type"),
    })

by_task = collections.defaultdict(list)
for row in rows:
    by_task[row["task_id"]].append(row)
for task_rows in by_task.values():
    task_rows.sort(key=lambda r: (r["source_priority"], r["init_state_id"], r["seed"], r["source_report"]))

selected = []
task_ids = list(range(10))
task_counts = collections.Counter()
while True:
    added = False
    for task in task_ids:
        if max_total > 0 and len(selected) >= max_total:
            break
        if task_counts[task] >= max_per_task:
            continue
        bucket = by_task.get(task) or []
        if task_counts[task] >= len(bucket):
            continue
        selected.append(bucket[task_counts[task]])
        task_counts[task] += 1
        added = True
    if not added or (max_total > 0 and len(selected) >= max_total):
        break

confirm = out / "confirm"
confirm.mkdir(parents=True, exist_ok=True)
summary = {
    "schema_version": "shed-cfs-v4-manytask-confirm-candidates-v1",
    "num_search_reports": len(list((out / "search").glob("gpu*/reports/*_causal_v4.json"))),
    "num_natural_failure_candidates": len(rows),
    "num_selected_for_k5": len(selected),
    "task_ids_with_candidates": sorted(by_task),
    "task_candidate_counts": {str(k): len(v) for k, v in sorted(by_task.items())},
    "max_confirm_candidates": max_total,
    "max_confirm_per_task": max_per_task,
    "candidates": selected,
}
(confirm / "candidate_index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with (confirm / "candidates.tsv").open("w", encoding="utf-8") as f:
    for row in selected:
        f.write("{task_id}\t{init_state_id}\t{seed}\t{source_priority}\t{source_report}\n".format(**row))
print(json.dumps({k: summary[k] for k in summary if k != "candidates"}, indent=2, ensure_ascii=False))
PY
}

split_confirm_candidates() {
  "$PY" - "$OUT" $GPUS <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
gpus = [str(x) for x in sys.argv[2:]]
rows = [line for line in (out / "confirm" / "candidates.tsv").read_text(encoding="utf-8").splitlines() if line.strip()]
for gpu in gpus:
    shard = out / "confirm" / f"gpu{gpu}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "reports").mkdir(exist_ok=True)
    (shard / "videos").mkdir(exist_ok=True)
    (shard / "logs").mkdir(exist_ok=True)
    (shard / "cases.tsv").write_text("", encoding="utf-8")
for idx, row in enumerate(rows):
    gpu = gpus[idx % len(gpus)]
    with (out / "confirm" / f"gpu{gpu}" / "cases.tsv").open("a", encoding="utf-8") as f:
        f.write(row + "\n")
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

start_confirm_policy_server() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/confirm/gpu${gpu}"
  if port_open "$port"; then
    log "ERROR confirm port $port already open before launch"
    return 3
  fi
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
  echo $! > "$shard/policy_server.pid"
  wait_port "$port"
}

run_confirm_shard() {
  local gpu="$1"
  local port=$((CONFIRM_PORT_BASE + gpu))
  local shard="$OUT/confirm/gpu${gpu}"
  mkdir -p "$shard"/{reports,videos,logs}
  echo "$$" > "$shard/shard.pid"
  start_confirm_policy_server "$gpu" "$port"
  while IFS=$'\t' read -r task init seed source_priority source_report; do
    if [[ -z "${task:-}" ]]; then
      continue
    fi
    local prefix
    prefix="$(printf 'task%02d_init%02d_seed%02d' "$task" "$init" "$seed")"
    local report="$shard/reports/${prefix}_causal_v4.json"
    local log_path="$shard/logs/${prefix}.log"
    if [[ -s "$report" ]]; then
      echo "$(date -Is) skip existing $prefix" | tee -a "$shard/shard.log"
      continue
    fi
    echo "$(date -Is) K5 start $prefix source_priority=$source_priority source=$source_report" | tee -a "$shard/shard.log"
    local cmd=(
      "$PY" "$CODE/pi05_natural_failure_probe.py"
      --policy-host 127.0.0.1
      --policy-port "$port"
      --policy-config pi0_fast_libero
      --policy-checkpoint "$POLICY_DIR"
      --task-suite-name libero_10
      --task-ids "$task"
      --init-state-ids "$init"
      --seed "$seed"
      --replay-trials 5
      --search-replay-trials 1
      --confirm-replay-trials 5
      --repair-replay-trials 5
      --event-window 32
      --causal-context-before 48
      --causal-context-after 8
      --causal-max-units 18
      --causal-ablation-trials 5
      --causal-ablation-strategies hold,adjacent,gripper_correction
      --repair-scheduling-mode pass_hunt
      --demo-repair-timeout-seconds 30
      --continuation recorded
      --camera-size 512
      --gpu-id "$gpu"
      --output "$report"
      --enable-visual-policy-mask
      --record-video
      --video-dir "$shard/videos"
      --video-prefix "$prefix"
      --video-camera agentview_image
      --video-fps 30
      --video-every-n 1
      --video-codec libx264
      --video-quality 10
      --progress-log-path "$shard/logs/${prefix}_progress.jsonl"
    )
    set +e
    if [[ "$CONFIRM_CASE_TIMEOUT_SECONDS" != "0" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        timeout "$CONFIRM_CASE_TIMEOUT_SECONDS" "${cmd[@]}" > "$log_path" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        "${cmd[@]}" > "$log_path" 2>&1
    fi
    local status=$?
    set -e
    echo "$(date -Is) K5 done $prefix status=$status" | tee -a "$shard/shard.log"
  done < "$shard/cases.tsv"
}

summarize_confirm_reports() {
  log "summarizing K=5 reports"
  "$PY" - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rows = []
for path in sorted((out / "confirm").glob("gpu*/reports/*_causal_v4.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append({"report_path": str(path), "status": "unreadable", "error": type(exc).__name__})
        continue
    selected = report.get("selected_failed_rollout") or {}
    causal = report.get("causal_validation") or {}
    row = {
        "report_path": str(path),
        "status": "ok",
        "task_id": selected.get("task_id"),
        "init_state_id": selected.get("init_state_id"),
        "seed": selected.get("reset_seed"),
        "natural_failure": selected.get("success") is False if selected else None,
        "same_failure_necessity_pass": bool(report.get("same_failure_necessity_pass") or causal.get("same_failure_necessity_pass")),
        "policy_strong_repair_valid_pass": bool(report.get("policy_strong_repair_valid_pass") or causal.get("policy_strong_repair_valid_pass")),
        "full_success_repair_pass": bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass")),
    }
    rows.append(row)
summary = {
    "schema_version": "shed-cfs-v4-manytask-k5-summary-v1",
    "num_reports": len(rows),
    "num_ok_reports": sum(1 for r in rows if r.get("status") == "ok"),
    "task_ids": sorted({r.get("task_id") for r in rows if isinstance(r.get("task_id"), int)}),
    "same_failure_necessity_pass": sum(1 for r in rows if r.get("same_failure_necessity_pass") is True),
    "policy_strong_repair_valid_pass": sum(1 for r in rows if r.get("policy_strong_repair_valid_pass") is True),
    "full_success_repair_pass": sum(1 for r in rows if r.get("full_success_repair_pass") is True),
    "reports": rows,
}
(out / "confirm" / "k5_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: summary[k] for k in summary if k != "reports"}, indent=2, ensure_ascii=False))
PY
}

split_review_reports() {
  "$PY" - "$OUT" $GPUS <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
gpus = [str(x) for x in sys.argv[2:]]
reports = []
for path in sorted((out / "confirm").glob("gpu*/reports/*_causal_v4.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    selected = report.get("selected_failed_rollout") or {}
    if not selected or selected.get("success") is not False:
        continue
    reports.append(path)
for gpu in gpus:
    shard = out / "review" / f"gpu{gpu}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "reports.txt").write_text("", encoding="utf-8")
for idx, report in enumerate(reports):
    gpu = gpus[idx % len(gpus)]
    with (out / "review" / f"gpu{gpu}" / "reports.txt").open("a", encoding="utf-8") as f:
        f.write(str(report) + "\n")
(out / "review" / "report_paths.txt").write_text("\n".join(str(p) for p in reports) + ("\n" if reports else ""), encoding="utf-8")
print(len(reports))
PY
}

start_review_shards() {
  log "starting recorded_error six-panel review shards"
  local pids=()
  local gpu
  for gpu in $GPUS; do
    local shard="$OUT/review/gpu${gpu}"
    mapfile -t reports < "$shard/reports.txt"
    if [[ "${#reports[@]}" -eq 0 ]]; then
      log "review skip gpu=$gpu no reports"
      continue
    fi
    local port=$((REVIEW_PORT_BASE + gpu))
    log "review launch gpu=$gpu port=$port reports=${#reports[@]}"
    POLICY_REPAIR_REVIEW_MODE=recorded_error \
      "$ROOT/v4/scripts/run_source_aware_review_shard_20260530.sh" \
        "$gpu" "$port" "$shard" "${reports[@]}" \
        > "$shard/master.log" 2>&1 &
    pids+=("$!")
  done
  printf "%s\n" "${pids[@]}" > "$OUT/review/shard.pids"
  wait_shards "review" "${pids[@]}" || true
}

summarize_review() {
  log "summarizing review package"
  "$PY" - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
summary = {
    "schema_version": "shed-cfs-v4-manytask-recorded-error-review-summary-v1",
    "output_dir": str(out),
    "review_indexes": [],
    "num_cases": 0,
    "num_multisource_videos": 0,
    "task_ids": [],
    "cases": [],
}
task_ids = set()
for manifest_path in sorted((out / "review").glob("gpu*/review/review_manifest.json")):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary["review_indexes"].append(str(manifest_path.parent / "review_index.html"))
    for case in data.get("cases", []):
        task_id = case.get("task_id")
        if isinstance(task_id, int):
            task_ids.add(task_id)
        videos = []
        for trial in case.get("trials") or []:
            video = trial.get("repair_multisource_video") or trial.get("repair_quadriptych_video")
            if video:
                videos.append(video)
        row = {
            "case_id": case.get("case_id"),
            "task_id": task_id,
            "init_state_id": case.get("init_state_id"),
            "seed": case.get("seed"),
            "failure_type": case.get("failure_type"),
            "case_review_path": case.get("case_review_path"),
            "review_index": str(manifest_path.parent / "review_index.html"),
            "multisource_videos": videos,
        }
        summary["cases"].append(row)
        summary["num_cases"] += 1
        summary["num_multisource_videos"] += len(videos)
summary["task_ids"] = sorted(task_ids)
(out / "review" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: summary[k] for k in summary if k != "cases"}, indent=2, ensure_ascii=False))
PY
}

main() {
  : > "$OUT/master.log"
  log "manytask recorded-error pipeline starting"
  log "OUT=$OUT"
  log "GPUS=$GPUS TASK_IDS=$TASK_IDS SEEDS=$SEEDS"
  require_ports_free "$SEARCH_PORT_BASE"
  require_ports_free "$CONFIRM_PORT_BASE"
  require_ports_free "$REVIEW_PORT_BASE"

  local search_pids=()
  local gpu
  for gpu in $GPUS; do
    start_search_shard "$gpu"
    search_pids+=("$(cat "$OUT/search/gpu${gpu}/run.pid")")
  done
  printf "%s\n" "${search_pids[@]}" > "$OUT/search/shard.pids"
  wait_shards "search" "${search_pids[@]}" || true

  select_confirm_candidates
  split_confirm_candidates

  local confirm_pids=()
  for gpu in $GPUS; do
    (
      run_confirm_shard "$gpu"
    ) > "$OUT/confirm/gpu${gpu}/master.log" 2>&1 &
    confirm_pids+=("$!")
  done
  printf "%s\n" "${confirm_pids[@]}" > "$OUT/confirm/shard.pids"
  wait_shards "confirm" "${confirm_pids[@]}" || true

  summarize_confirm_reports
  split_review_reports
  start_review_shards
  summarize_review
  log "manytask recorded-error pipeline finished"
}

main "$@"
