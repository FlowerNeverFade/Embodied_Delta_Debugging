#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/yanghaoyun/research/Embodied_Delta_Debugging"
CODE="$ROOT/v4/code"
PY="/data2/yanghaoyun/envs/libero38/bin/python"
OUT="$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530"
NULL_REPORT_DIR="$OUT/nonexistent_report_dir"
POLICY_DIR="$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay"
ACTION_TOKENIZER="$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean"
TEXT_TOKENIZER="/data2/yanghaoyun/research/VLA_SKILL/model/google/paligemma-3b-pt-224"
DEMO_ROOT="/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"

export EDD_PROJECT_ROOT="$ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero_config"
export OPENPI_PYTHON="$PY"
export LEROBOT_PYTHON="/data2/yanghaoyun/miniconda3/bin/python"
export OPENPI_CLIENT_SRC="/data2/yanghaoyun/research/openpi/packages/openpi-client/src"
export PYTHONPATH="$CODE:$OPENPI_CLIENT_SRC:/data2/yanghaoyun/research/LIBERO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"

REPORT_BASE="$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528"

declare -A REPORTS
REPORTS[gpu0]="$REPORT_BASE/shards/gpu0/reports/task01_init01_seed07_causal_v4.json $REPORT_BASE/shards/gpu1/reports/task01_init15_seed07_causal_v4.json"
REPORTS[gpu1]="$REPORT_BASE/shards/gpu2/reports/task08_init46_seed17_causal_v4.json $REPORT_BASE/shards/gpu2/reports/task08_init18_seed07_causal_v4.json"
REPORTS[gpu2]="$REPORT_BASE/shards/gpu0/reports/task08_init21_seed17_causal_v4.json $REPORT_BASE/shards/gpu3/reports/task05_init44_seed77_causal_v4.json"
REPORTS[gpu3]="$REPORT_BASE/shards/gpu1/reports/task02_init47_seed17_causal_v4.json $REPORT_BASE/shards/gpu1/reports/task09_init38_seed07_causal_v4.json"

run_shard() {
  local gpu="$1"
  local port="$2"
  local shard="$OUT/gpu${gpu}"
  mkdir -p "$shard"
  local args=()
  for report in ${REPORTS[gpu${gpu}]}; do
    if [[ ! -f "$report" ]]; then
      echo "missing report: $report" >&2
      return 2
    fi
    args+=(--report-path "$report")
  done
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID="$gpu" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
      -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    "$PY" "$CODE/slice_review_export.py" \
      "${args[@]}" \
      --report-dir "$NULL_REPORT_DIR" \
      --output-dir "$shard/review" \
      --include-necessity-only \
      --replay-trials-to-record 1 \
      --policy-repair-review-mode "${POLICY_REPAIR_REVIEW_MODE:-recorded_error}" \
      --review-top-k-sets 1 \
      --task-suite-name libero_10 \
      --policy-port "$port" \
      --launch-policy-server \
      --cuda-visible-devices "$gpu" \
      --policy-dir "$POLICY_DIR" \
      --action-tokenizer-path "$ACTION_TOKENIZER" \
      --text-tokenizer-path "$TEXT_TOKENIZER" \
      --demo-dataset-root "$DEMO_ROOT" \
      --camera-size 512 \
      --resize-size 224 \
      --video-fps 30 \
      --video-codec libx264 \
      --video-quality 10 \
      > "$shard/review_export.log" 2>&1
}

echo "$(date -Is) source-aware top-candidate review starting" | tee "$OUT/master.log"
echo "$$" > "$OUT/run.pid"

declare -a pids=()
for gpu in 0 1 2 3; do
  port=$((8070 + gpu))
  echo "$(date -Is) launching gpu=$gpu port=$port" | tee -a "$OUT/master.log"
  run_shard "$gpu" "$port" &
  pids+=("$!")
done
printf "%s\n" "${pids[@]}" > "$OUT/shard.pids"
echo "$(date -Is) shard_pids=${pids[*]}" | tee -a "$OUT/master.log"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

"$PY" - <<'PY' "$OUT" || true
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "schema_version": "source-aware-top-candidates-summary-v1",
    "output_dir": str(out),
    "shards": [],
    "num_cases": 0,
    "num_recorded_success": 0,
    "num_recorded_improvement": 0,
    "num_reported_vs_recorded_mismatch": 0,
}
for manifest_path in sorted(out.glob("gpu*/review/review_manifest.json")):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = {
        "manifest_path": str(manifest_path),
        "review_index": str(manifest_path.parent / "review_index.html"),
        "num_cases": data.get("num_cases", 0),
        "cases": [],
    }
    for case in data.get("cases", []):
        evidence = case.get("recorded_repair_evidence") or {}
        row = {
            "case_id": case.get("case_id"),
            "task_id": case.get("task_id"),
            "init_state_id": case.get("init_state_id"),
            "seed": case.get("seed"),
            "failure_type": case.get("failure_type"),
            "reported_full_success_repair_pass": case.get("full_success_repair_pass"),
            "recorded_any_success": evidence.get("any_success"),
            "recorded_any_improvement": evidence.get("any_improvement"),
            "reported_vs_recorded_mismatch": evidence.get("reported_vs_recorded_mismatch"),
            "case_review_path": case.get("case_review_path"),
        }
        shard["cases"].append(row)
        summary["num_cases"] += 1
        if row["recorded_any_success"] is True:
            summary["num_recorded_success"] += 1
        if row["recorded_any_improvement"] is True:
            summary["num_recorded_improvement"] += 1
        if row["reported_vs_recorded_mismatch"] is True:
            summary["num_reported_vs_recorded_mismatch"] += 1
    summary["shards"].append(shard)
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "$(date -Is) source-aware top-candidate review finished status=$status" | tee -a "$OUT/master.log"
exit "$status"
