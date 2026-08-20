#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

BASE_OUTPUT_DIR="${MULTIGPU_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_long_breakfast_video_causal_v2_multigpu_20260526}"
GPU_IDS=(${MULTIGPU_GPU_IDS:-0 1 2})
PORTS=(${MULTIGPU_POLICY_PORTS:-8030 8031 8032})

echo "multigpu_output=$BASE_OUTPUT_DIR"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
echo

for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  port="${PORTS[$i]}"
  shard_output="$BASE_OUTPUT_DIR/gpu${gpu}"
  echo "===== shard=$i gpu=$gpu port=$port ====="
  ULTRA_OUTPUT_DIR="$shard_output" POLICY_PORT="$port" "$RUN_DIR/status_ultra_video_eval.sh" || true
  echo
done

/root/autodl-tmp/envs/libero38/bin/python - "$BASE_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
rows = []
for manifest in sorted(base.glob("gpu*/manifest.jsonl")):
    for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
print("===== aggregate =====")
print(f"manifest_lines={len(rows)}")
print(f"reports={len(list(base.glob('gpu*/reports/*_causal_v*.json')))}")
print(f"videos={len(list(base.glob('gpu*/videos/*.mp4')))}")
for key in ("semantic_pass", "semantic_nonpass", "probe_failed", "timeout", "skipped_existing"):
    print(f"{key}={sum(1 for r in rows if r.get('status') == key)}")
print(f"natural_failures={sum(1 for r in rows if r.get('natural_failure_found') is True)}")
print(f"same_failure_passes={sum(1 for r in rows if r.get('same_failure_pass') is True)}")
print(f"causal_passes={sum(1 for r in rows if r.get('causal_pass') is True)}")
print(f"positive_windows={sum(int(r.get('positive_windows') or 0) for r in rows)}")
PY
