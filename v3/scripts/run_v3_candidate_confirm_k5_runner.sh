#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
CODE_ROOT="$PROJECT_ROOT/v3/code_snapshot"
source "$RUN_DIR/config.env"

CONFIRM_OUTPUT_DIR="${CONFIRM_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_confirm_k5_runner_20260527}"
CONFIRM_GPU_IDS=(${CONFIRM_GPU_IDS:-0 1 2})
CONFIRM_POLICY_PORTS=(${CONFIRM_POLICY_PORTS:-8030 8031 8032})
CONFIRM_CASE_TIMEOUT_SECONDS="${CONFIRM_CASE_TIMEOUT_SECONDS:-0}"
CONFIRM_REPAIR_REPLAY_TRIALS="${CONFIRM_REPAIR_REPLAY_TRIALS:-5}"
CONFIRM_CAUSAL_MAX_UNITS="${CONFIRM_CAUSAL_MAX_UNITS:-4}"
CONFIRM_CAUSAL_CONTEXT_BEFORE="${CONFIRM_CAUSAL_CONTEXT_BEFORE:-48}"
CONFIRM_CAUSAL_CONTEXT_AFTER="${CONFIRM_CAUSAL_CONTEXT_AFTER:-8}"
CONFIRM_CAMERA_SIZE="${CONFIRM_CAMERA_SIZE:-512}"
CONFIRM_VIDEO_FPS="${CONFIRM_VIDEO_FPS:-30}"
CONFIRM_VIDEO_QUALITY="${CONFIRM_VIDEO_QUALITY:-10}"
CONFIRM_VIDEO_CODEC="${CONFIRM_VIDEO_CODEC:-libx264}"
CONFIRM_VIDEO_CAMERA="${CONFIRM_VIDEO_CAMERA:-agentview_image}"
CONFIRM_XLA_MEM_FRACTION="${CONFIRM_XLA_MEM_FRACTION:-0.55}"
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
    if raw:
        task_id, init_state_id, seed = (int(x) for x in raw.split(","))
        cases.append(
            {"task_id": task_id, "init_state_id": init_state_id, "seed": seed}
        )
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
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
PY

run_case() {
  local gpu="$1"
  local port="$2"
  local task_id="$3"
  local init_state_id="$4"
  local seed="$5"
  local case_name="task$(printf "%02d" "$task_id")_init$(printf "%02d" "$init_state_id")_seed$(printf "%02d" "$seed")"
  local case_dir="$CONFIRM_OUTPUT_DIR/gpu${gpu}/${case_name}"
  mkdir -p "$case_dir"
  echo "[$(date -Is)] start $case_name gpu=$gpu port=$port"
  set +e
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
      -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    /root/autodl-tmp/envs/libero38/bin/python -u "$CODE_ROOT/run_risk_critic_large_eval.py" \
      --output-dir "$case_dir" \
      --report-dir "$case_dir/reports" \
      --log-dir "$case_dir/logs" \
      --manifest-path "$case_dir/manifest.jsonl" \
      --cost-summary-path "$case_dir/cost_summary.json" \
      --export-path "$case_dir/risk_critic_full_v1.jsonl" \
      --train-output "$case_dir/risk_critic_full_metrics.json" \
      --summary-path "$case_dir/summary.json" \
      --policy-host 127.0.0.1 \
      --policy-port "$port" \
      --launch-policy-server \
      --policy-server-kind "$POLICY_SERVER_KIND" \
      --policy-config "$POLICY_CONFIG" \
      --policy-dir "$POLICY_DIR" \
      --action-tokenizer-path "$ACTION_TOKENIZER_PATH" \
      --text-tokenizer-path "$TEXT_TOKENIZER_PATH" \
      --pytorch-device "$PYTORCH_DEVICE" \
      --pytorch-compile-mode "$PYTORCH_COMPILE_MODE" \
      --task-suite-name "$DATASET_NAME" \
      --task-ids "$task_id" \
      --init-state-ids "$init_state_id" \
      --seeds "$seed" \
      --max-cases 0 \
      --positive-target 0 \
      --min-cases-before-positive-stop 0 \
      --cuda-visible-devices "$gpu" \
      --xla-mem-fraction "$CONFIRM_XLA_MEM_FRACTION" \
      --search-replay-trials 1 \
      --confirm-replay-trials 5 \
      --repair-replay-trials "$CONFIRM_REPAIR_REPLAY_TRIALS" \
      --replay-trials 5 \
      --continuation recorded \
      --event-window 32 \
      --causal-context-before "$CONFIRM_CAUSAL_CONTEXT_BEFORE" \
      --causal-context-after "$CONFIRM_CAUSAL_CONTEXT_AFTER" \
      --causal-max-units "$CONFIRM_CAUSAL_MAX_UNITS" \
      --causal-ablation-trials 5 \
      --case-timeout-seconds "$CONFIRM_CASE_TIMEOUT_SECONDS" \
      --train-steps 1 \
      --record-video \
      --require-video \
      --video-dir "$case_dir/videos" \
      --video-camera "$CONFIRM_VIDEO_CAMERA" \
      --video-fps "$CONFIRM_VIDEO_FPS" \
      --video-every-n 1 \
      --video-codec "$CONFIRM_VIDEO_CODEC" \
      --video-quality "$CONFIRM_VIDEO_QUALITY" \
      --camera-size "$CONFIRM_CAMERA_SIZE" \
      --enable-visual-policy-mask \
      --skip-source-repair-if-policy-pass \
      --stop-after-first-repair-valid-core
  local rc="$?"
  set -e
  echo "[$(date -Is)] finish $case_name gpu=$gpu rc=$rc"
  return "$rc"
}

worker() {
  local worker_idx="$1"
  local gpu="${CONFIRM_GPU_IDS[$worker_idx]}"
  local port="${CONFIRM_POLICY_PORTS[$worker_idx]}"
  IFS=';' read -r -a cases <<< "$CONFIRM_CANDIDATES"
  for idx in "${!cases[@]}"; do
    if (( idx % ${#CONFIRM_GPU_IDS[@]} != worker_idx )); then
      continue
    fi
    IFS=',' read -r task_id init_state_id seed <<< "${cases[$idx]}"
    run_case "$gpu" "$port" "$task_id" "$init_state_id" "$seed"
  done
}

for i in "${!CONFIRM_GPU_IDS[@]}"; do
  worker "$i" > "$CONFIRM_OUTPUT_DIR/gpu${CONFIRM_GPU_IDS[$i]}.worker.log" 2>&1 &
  echo "$!" > "$CONFIRM_OUTPUT_DIR/gpu${CONFIRM_GPU_IDS[$i]}.worker.pid"
done

wait

python3 - "$CONFIRM_OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for summary_path in sorted(out.glob("gpu*/task*/summary.json")):
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for row in summary.get("rows") or []:
        row["case_summary_path"] = str(summary_path)
        rows.append(row)

summary = {
    "schema_version": "shed-cfs-causal-v3-confirm-summary-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(out),
    "processed_cases": len(rows),
    "status_counts": dict(Counter(row.get("status") for row in rows)),
    "same_failure_passes": sum(row.get("same_failure_pass") is True for row in rows),
    "policy_strong_repair_valid_passes": sum(
        row.get("policy_strong_repair_valid_pass") is True for row in rows
    ),
    "full_success_repair_passes": sum(
        row.get("full_success_repair_pass") is True for row in rows
    ),
    "positive_windows": sum(int(row.get("positive_windows") or 0) for row in rows),
    "rows": rows,
}
(out / "confirm_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
