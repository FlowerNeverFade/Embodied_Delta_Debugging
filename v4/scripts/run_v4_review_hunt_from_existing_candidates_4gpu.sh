#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V4="$ROOT/v4"
CODE="$V4/code"
PY="/root/autodl-tmp/envs/libero38/bin/python"
LEROBOT_PY="/root/miniconda3/bin/python"
OUT="${OUT:-$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528}"
POLICY_DIR="$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay"
ACTION_TOKENIZER="$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean"
TEXT_TOKENIZER="/root/autodl-tmp/research/VLA_SKILL/model/google/paligemma-3b-pt-224"
PORT_BASE="${PORT_BASE:-8070}"
GPUS="${GPUS:-0 1 2 3}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"

mkdir -p "$OUT"/{logs,shards,review}
ln -sfn "$OUT" "$V4/outputs/review_hunt_k5_latest"

port_open() {
  local port="$1"
  "$PY" - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
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

generate_candidates() {
  "$PY" - "$OUT" "$MAX_CANDIDATES" "$ROOT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
max_candidates = int(sys.argv[2])
root = pathlib.Path(sys.argv[3]) / "model_datasets/pi0fast-libero-libero_10/outputs"
items = {}
for path in root.rglob("*_causal_v*.json"):
    if "/reports/" not in str(path):
        continue
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    schema = str(report.get("schema_version", ""))
    if "causal-v3" not in schema and "causal-v4" not in schema:
        continue
    selected = report.get("selected_failed_rollout") or {}
    task = selected.get("task_id")
    init = selected.get("init_state_id")
    seed = selected.get("reset_seed")
    if task is None or init is None or seed is None:
        continue
    causal = report.get("causal_validation") or {}
    same = bool(report.get("same_failure_necessity_pass") or causal.get("same_failure_necessity_pass"))
    policy = bool(report.get("policy_strong_repair_valid_pass") or causal.get("policy_strong_repair_valid_pass"))
    full = bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass"))
    if not (same or policy or full):
        continue
    key = (int(task), int(init), int(seed))
    priority = 0 if (policy or full) else 1
    if "causal-v4" in schema and (policy or full):
        priority = -1
    score = (priority, 0 if full else 1, str(path))
    prev = items.get(key)
    if prev is None or score < prev["score"]:
        items[key] = {
            "score": score,
            "task_id": key[0],
            "init_state_id": key[1],
            "seed": key[2],
            "source_report": str(path),
            "source_schema": schema,
            "source_same_failure_necessity_pass": same,
            "source_policy_strong_repair_valid_pass": policy,
            "source_full_success_repair_pass": full,
        }

rows = sorted(items.values(), key=lambda item: (item["score"], item["task_id"], item["init_state_id"], item["seed"]))
if max_candidates > 0:
    rows = rows[:max_candidates]
out.mkdir(parents=True, exist_ok=True)
(out / "candidate_index.json").write_text(
    json.dumps(
        {
            "schema_version": "shed-cfs-v4-review-hunt-candidates-v1",
            "num_candidates": len(rows),
            "candidates": rows,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
with (out / "candidates.tsv").open("w", encoding="utf-8") as f:
    for item in rows:
        f.write(
            "{task_id}\t{init_state_id}\t{seed}\t{source_schema}\t{source_report}\n".format(**item)
        )
print(len(rows))
PY
}

split_candidates() {
  "$PY" - "$OUT" $GPUS <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
gpus = [str(x) for x in sys.argv[2:]]
rows = [line for line in (out / "candidates.tsv").read_text(encoding="utf-8").splitlines() if line.strip()]
for gpu in gpus:
    shard = out / "shards" / f"gpu{gpu}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "reports").mkdir(exist_ok=True)
    (shard / "videos").mkdir(exist_ok=True)
    (shard / "logs").mkdir(exist_ok=True)
    (shard / "cases.tsv").write_text("", encoding="utf-8")
for idx, line in enumerate(rows):
    gpu = gpus[idx % len(gpus)]
    shard = out / "shards" / f"gpu{gpu}"
    with (shard / "cases.tsv").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
PY
}

start_policy_server() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/shards/gpu$gpu"
  if port_open "$port"; then
    echo "Port $port already open; reusing existing policy server for gpu $gpu" | tee -a "$OUT/master.log"
    return 0
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

run_shard() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/shards/gpu$gpu"
  start_policy_server "$gpu" "$port"
  while IFS=$'\t' read -r task init seed source_schema source_report; do
    if [[ -z "${task:-}" ]]; then
      continue
    fi
    local prefix
    prefix="$(printf 'task%02d_init%02d_seed%02d' "$task" "$init" "$seed")"
    local report="$shard/reports/${prefix}_causal_v4.json"
    local log="$shard/logs/${prefix}.log"
    if [[ -s "$report" ]]; then
      echo "$(date -Is) gpu=$gpu skip existing $prefix" | tee -a "$shard/shard.log"
      continue
    fi
    echo "$(date -Is) gpu=$gpu start $prefix source=$source_schema" | tee -a "$shard/shard.log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH="$CODE" \
      "$PY" "$CODE/pi05_natural_failure_probe.py" \
        --policy-host 127.0.0.1 \
        --policy-port "$port" \
        --policy-config pi0_fast_libero \
        --policy-checkpoint "$POLICY_DIR" \
        --task-suite-name libero_10 \
        --task-ids "$task" \
        --init-state-ids "$init" \
        --seed "$seed" \
        --replay-trials 5 \
        --search-replay-trials 1 \
        --confirm-replay-trials 5 \
        --repair-replay-trials 5 \
        --event-window 32 \
        --causal-context-before 48 \
        --causal-context-after 8 \
        --causal-max-units 18 \
        --causal-ablation-trials 5 \
        --causal-ablation-strategies hold,adjacent,gripper_correction \
        --repair-scheduling-mode pass_hunt \
        --demo-repair-timeout-seconds 30 \
        --continuation recorded \
        --camera-size 512 \
        --gpu-id "$gpu" \
        --output "$report" \
        --enable-visual-policy-mask \
        --record-video \
        --video-dir "$shard/videos" \
        --video-prefix "$prefix" \
        --video-camera agentview_image \
        --video-fps 30 \
        --video-every-n 1 \
        --video-codec libx264 \
        --video-quality 10 \
        --progress-log-path "$shard/logs/${prefix}_progress.jsonl" \
        > "$log" 2>&1 || true
    echo "$(date -Is) gpu=$gpu done $prefix" | tee -a "$shard/shard.log"
  done < "$shard/cases.tsv"
}

summarize_reports() {
  "$PY" - "$OUT" <<'PY'
import json
import pathlib
import statistics
import collections
import sys

out = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out.glob("shards/gpu*/reports/*_causal_v4.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append({"report_path": str(path), "status": "unreadable", "error": type(exc).__name__})
        continue
    causal = report.get("causal_validation") or {}
    selected = report.get("selected_failed_rollout") or {}
    row = {
        "report_path": str(path),
        "status": "ok",
        "task_id": selected.get("task_id"),
        "init_state_id": selected.get("init_state_id"),
        "seed": selected.get("reset_seed"),
        "natural_failure": bool(selected and not selected.get("success", True)),
        "same_failure": bool((report.get("reproduction_statistics") or {}).get("same_failure")),
        "same_failure_necessity_pass": bool(report.get("same_failure_necessity_pass") or causal.get("same_failure_necessity_pass")),
        "repair_valid_causal_pass": bool(report.get("repair_valid_causal_pass") or causal.get("repair_valid_causal_pass")),
        "policy_strong_repair_valid_pass": bool(report.get("policy_strong_repair_valid_pass") or causal.get("policy_strong_repair_valid_pass")),
        "full_success_repair_pass": bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass")),
        "raw_policy_repair_valid_pass": bool(report.get("policy_raw_repair_valid_pass") or causal.get("raw_policy_repair_valid_pass")),
        "language_phrase_repair_valid_pass": bool(report.get("policy_language_phrase_repair_valid_pass") or causal.get("language_phrase_repair_valid_pass")),
        "visual_mask_repair_valid_pass": bool(report.get("policy_visual_mask_repair_valid_pass") or causal.get("visual_mask_repair_valid_pass")),
        "demo_existence_repair_pass": bool(report.get("demo_existence_repair_pass") or causal.get("demo_existence_repair_pass")),
        "cost_seconds": (report.get("cost_summary") or {}).get("total_wall_seconds"),
        "video_path": (selected or {}).get("video_path"),
    }
    rows.append(row)
counts = collections.Counter()
for row in rows:
    for key in [
        "natural_failure",
        "same_failure",
        "same_failure_necessity_pass",
        "repair_valid_causal_pass",
        "policy_strong_repair_valid_pass",
        "full_success_repair_pass",
        "raw_policy_repair_valid_pass",
        "language_phrase_repair_valid_pass",
        "visual_mask_repair_valid_pass",
        "demo_existence_repair_pass",
    ]:
        if row.get(key) is True:
            counts[key] += 1
costs = [row["cost_seconds"] for row in rows if isinstance(row.get("cost_seconds"), (int, float))]
review_candidates = [
    row["report_path"]
    for row in rows
    if row.get("repair_valid_causal_pass") or row.get("same_failure_necessity_pass")
]
review_candidates = review_candidates[:12]
(out / "review_report_paths.txt").write_text("\n".join(review_candidates) + ("\n" if review_candidates else ""), encoding="utf-8")
summary = {
    "schema_version": "shed-cfs-v4-review-hunt-summary-v1",
    "num_reports": len(rows),
    "counts": dict(counts),
    "cost_seconds_median": None if not costs else statistics.median(costs),
    "cost_seconds_mean": None if not costs else statistics.mean(costs),
    "review_candidate_reports": review_candidates,
    "reports": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"num_reports": len(rows), "counts": dict(counts), "review_candidates": len(review_candidates)}, ensure_ascii=False))
PY
}

export_review_pack() {
  summarize_reports
  local reports_file="$OUT/review_report_paths.txt"
  if [[ ! -s "$reports_file" ]]; then
    echo "No v4 repair/necessity reports available for review yet." | tee -a "$OUT/master.log"
    return 0
  fi
  local args=()
  while IFS= read -r report; do
    [[ -n "$report" ]] && args+=(--report-path "$report")
  done < "$reports_file"
  CUDA_VISIBLE_DEVICES=0 \
  MUJOCO_EGL_DEVICE_ID=0 \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  PYTHONPATH="$CODE" \
    "$PY" "$CODE/slice_review_export.py" \
      "${args[@]}" \
      --output-dir "$OUT/review" \
      --include-necessity-only \
      --max-cases 12 \
      --replay-trials-to-record 5 \
      --review-top-k-sets 5 \
      --task-suite-name libero_10 \
      --policy-port "$PORT_BASE" \
      --allow-existing-policy-server \
      --camera-size 512 \
      --video-fps 30 \
      --video-codec libx264 \
      --video-quality 10 \
      > "$OUT/review_export.log" 2>&1 || true
}

echo "$(date -Is) generating candidates" | tee "$OUT/master.log"
num_candidates="$(generate_candidates)"
echo "$(date -Is) candidates=$num_candidates" | tee -a "$OUT/master.log"
split_candidates

declare -a pids=()
idx=0
for gpu in $GPUS; do
  port=$((PORT_BASE + idx))
  run_shard "$gpu" "$port" > "$OUT/shards/gpu$gpu/runner.log" 2>&1 &
  pids+=("$!")
  idx=$((idx + 1))
done
printf '%s\n' "${pids[@]}" > "$OUT/shard_runner.pids"
echo "$(date -Is) shard_pids=${pids[*]}" | tee -a "$OUT/master.log"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "$(date -Is) shards finished status=$status" | tee -a "$OUT/master.log"
export_review_pack
echo "$(date -Is) review export finished" | tee -a "$OUT/master.log"

exit "$status"
