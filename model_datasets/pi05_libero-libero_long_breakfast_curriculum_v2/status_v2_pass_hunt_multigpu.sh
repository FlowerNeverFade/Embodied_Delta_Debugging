#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${V2_PASS_HUNT_OUTPUT_DIR:-$RUN_DIR/outputs/v2_pass_hunt_20260526}"

echo "output=$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/run.pid" ]]; then
  pid="$(cat "$OUTPUT_DIR/run.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "runner_pid=$pid running=true"
  else
    echo "runner_pid=$pid running=false"
  fi
else
  echo "runner_pid=missing"
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
echo

/root/autodl-tmp/envs/libero38/bin/python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for manifest in [out / "manifest.jsonl", *sorted(out.glob("gpu*/manifest.jsonl"))]:
    if not manifest.exists():
        continue
    for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
print(f"manifest_lines={len(rows)}")
print(f"reports={len(list(out.glob('gpu*/reports/*_causal_v*.json')))}")
print(f"videos={len(list(out.glob('gpu*/videos/*.mp4')))}")
for key in ("semantic_pass", "semantic_nonpass", "probe_failed", "timeout", "skipped_existing"):
    print(f"{key}={sum(1 for row in rows if row.get('status') == key)}")
print(f"same_failure_pass={sum(1 for row in rows if row.get('same_failure_pass') is True)}")
print(f"causal_pass={sum(1 for row in rows if row.get('causal_pass') is True)}")
print(f"full_success_repair_pass={sum(1 for row in rows if row.get('full_success_repair_pass') is True)}")
print(f"positive_windows={sum(int(row.get('positive_windows') or 0) for row in rows)}")
summary = out / "summary.json"
if summary.exists():
    payload = json.loads(summary.read_text(encoding="utf-8"))
    print("summary_aggregate=" + json.dumps(payload.get("aggregate", {}), ensure_ascii=False))
PY
