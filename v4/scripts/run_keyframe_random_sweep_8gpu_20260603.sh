#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="${ROOT:-/data2/yanghaoyun/research/Embodied_Delta_Debugging}"
CODE="$ROOT/v4/code"
PY="${PY:-/data2/yanghaoyun/envs/libero38/bin/python}"
LEROBOT_PY="${LEROBOT_PY:-/data2/yanghaoyun/miniconda3/bin/python}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-/data2/yanghaoyun/research/openpi/packages/openpi-client/src}"
LIBERO_SRC="${LIBERO_SRC:-/data2/yanghaoyun/research/LIBERO}"
OUT="${OUT:-$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/keyframe_random_sweep_pi0fast_libero10_20260603}"

POLICY_DIR="${POLICY_DIR:-$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay}"
ACTION_TOKENIZER="${ACTION_TOKENIZER:-$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-/data2/yanghaoyun/research/VLA_SKILL/model/google/paligemma-3b-pt-224}"
MANIFEST="${MANIFEST:-$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/showcase_strict_success_by_task_20260531/manifest.json}"
DEMO_ROOT="${DEMO_ROOT:-/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PORT_BASE="${PORT_BASE:-8230}"
KEYFRAMES_PER_CASE="${KEYFRAMES_PER_CASE:-21}"
TRIALS="${TRIALS:-5}"
RANDOM_SEED="${RANDOM_SEED:-20260603}"
MAX_CASES="${MAX_CASES:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

export EDD_PROJECT_ROOT="$ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero_config"
export OPENPI_PYTHON="$PY"
export LEROBOT_PYTHON="$LEROBOT_PY"
export OPENPI_CLIENT_SRC
export PYTHONPATH="$CODE:$OPENPI_CLIENT_SRC:$LIBERO_SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"/{logs,shards}
ln -sfn "$OUT" "$ROOT/v4/outputs/keyframe_random_sweep_latest"
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

start_policy_server() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/shards/gpu${gpu}"
  mkdir -p "$shard"
  if port_open "$port"; then
    log "ERROR port $port already open; refusing to reuse an unverified policy server"
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

split_cases() {
  "$PY" - "$MANIFEST" "$OUT" $GPUS <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
gpus = [str(x) for x in sys.argv[3:]]
data = json.loads(manifest.read_text(encoding="utf-8"))
cases = [str(item["case_id"]) for item in data.get("cases", [])]
for gpu in gpus:
    shard = out / "shards" / f"gpu{gpu}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "case_ids.txt").write_text("", encoding="utf-8")
for idx, case_id in enumerate(cases):
    gpu = gpus[idx % len(gpus)]
    with (out / "shards" / f"gpu{gpu}" / "case_ids.txt").open("a", encoding="utf-8") as f:
        f.write(case_id + "\n")
print(json.dumps({"num_cases": len(cases), "gpus": gpus}, indent=2))
PY
}

run_shard() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/shards/gpu${gpu}"
  mapfile -t case_ids < "$shard/case_ids.txt"
  if [[ "${#case_ids[@]}" -eq 0 ]]; then
    log "shard gpu=$gpu skipped no cases"
    return 0
  fi
  local case_csv
  case_csv="$(IFS=,; echo "${case_ids[*]}")"
  if [[ "$SMOKE" == "1" ]]; then
    case_csv="task08_init21_seed17"
  fi
  local max_cases="$MAX_CASES"
  if [[ "$SMOKE" == "1" ]]; then
    max_cases="1"
  fi
  local keyframes="$KEYFRAMES_PER_CASE"
  local trials="$TRIALS"
  if [[ "$SMOKE" == "1" ]]; then
    keyframes="2"
    trials="1"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    start_policy_server "$gpu" "$port"
  fi
  local dry_args=()
  if [[ "$DRY_RUN" == "1" ]]; then
    dry_args+=(--dry-run)
  fi
  log "sweep launch gpu=$gpu port=$port cases=$case_csv keyframes=$keyframes trials=$trials dry_run=$DRY_RUN"
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
    "$PY" "$CODE/keyframe_random_sweep.py" \
      --manifest "$MANIFEST" \
      --output-dir "$shard" \
      --case-ids "$case_csv" \
      --max-cases "$max_cases" \
      --policy-port "$port" \
      --policy-dir "$POLICY_DIR" \
      --demo-dataset-root "$DEMO_ROOT" \
      --gpu-id "$gpu" \
      --keyframes-per-case "$keyframes" \
      --trials "$trials" \
      --random-seed "$RANDOM_SEED" \
      "${dry_args[@]}" \
      > "$shard/sweep.log" 2>&1
}

merge_outputs() {
  log "merging shard outputs"
  "$PY" - "$OUT" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
all_trials = []
for path in sorted(out.glob("shards/gpu*/keyframe_sweep_results.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_trials.append(json.loads(line))
(out / "keyframe_sweep_results.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_trials),
    encoding="utf-8",
)

all_summary = []
for path in sorted(out.glob("shards/gpu*/keyframe_success_summary.json")):
    all_summary.extend(json.loads(path.read_text(encoding="utf-8")))
(out / "keyframe_success_summary.json").write_text(
    json.dumps(all_summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
fields = [
    "case_id",
    "task_id",
    "init_state_id",
    "seed",
    "evidence_level",
    "keyframe",
    "keyframe_offset_from_repair_start",
    "repair_context_start",
    "repair_context_end",
    "minimal_start",
    "minimal_end",
    "planned_trials",
    "executed_trials",
    "success_count",
    "success_rate",
    "archive_source",
]
with (out / "keyframe_success_summary.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in all_summary:
        writer.writerow({key: row.get(key) for key in fields})

summary = {
    "schema_version": "keyframe-random-sweep-merged-summary-v1",
    "num_trials": len(all_trials),
    "num_keyframes": len(all_summary),
    "num_cases": len({row.get("case_id") for row in all_summary}),
    "case_ids": sorted({str(row.get("case_id")) for row in all_summary}),
    "mean_success_rate": (
        sum(float(row.get("success_rate", 0.0)) for row in all_summary) / len(all_summary)
        if all_summary else None
    ),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
  "$PY" "$CODE/keyframe_random_sweep.py" \
    --output-dir "$OUT" \
    --plot-only \
    --summary-json "$OUT/keyframe_success_summary.json" \
    > "$OUT/logs/plot.log" 2>&1
}

main() {
  log "keyframe random sweep starting OUT=$OUT GPUS=$GPUS DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
  split_cases | tee -a "$OUT/master.log"
  local pids=()
  local gpu
  for gpu in $GPUS; do
    (
      run_shard "$gpu" $((PORT_BASE + gpu))
    ) > "$OUT/shards/gpu${gpu}/master.log" 2>&1 &
    pids+=("$!")
  done
  printf "%s\n" "${pids[@]}" > "$OUT/shard.pids"
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  log "shards finished status=$status"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run finished"
    exit "$status"
  fi
  merge_outputs | tee -a "$OUT/master.log"
  log "keyframe random sweep finished"
  exit "$status"
}

main "$@"
