#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
OUTPUT_DIR="${ULTRA_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_targeted_k1_20260527}"
PID_FILE="$OUTPUT_DIR/run.pid"

echo "output_dir=$OUTPUT_DIR"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,ppid,stat,sid,etime,cmd || true
else
  echo "no pid file"
fi

/root/autodl-tmp/envs/libero38/bin/python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
manifest = out / "manifest.jsonl"
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
print(f"manifest_lines={len(rows)}")
print(f"reports={len(list((out / 'reports').glob('*_causal_v*.json')))}")
print(f"videos={len(list((out / 'videos').glob('*.mp4')))}")
print(f"semantic_pass={sum(1 for r in rows if r.get('status') == 'semantic_pass')}")
print(f"semantic_nonpass={sum(1 for r in rows if r.get('status') == 'semantic_nonpass')}")
print(f"probe_failed={sum(1 for r in rows if r.get('status') == 'probe_failed')}")
print(f"timeout={sum(1 for r in rows if r.get('status') == 'timeout')}")
print(f"positive_windows={sum(int(r.get('positive_windows') or 0) for r in rows)}")
print(f"video_exists_rows={sum(1 for r in rows if r.get('video_exists'))}")
print(f"summary_exists={int((out / 'summary.json').exists())}")
PY

tail -n 8 "$OUTPUT_DIR/manifest.jsonl" 2>/dev/null || true
