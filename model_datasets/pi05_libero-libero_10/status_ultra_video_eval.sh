#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
OUTPUT_DIR="${ULTRA_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_20260525}"
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
manifest = out / "manifest.jsonl"
rows = []
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
reports = list((out / "reports").glob("*_causal_v1.json"))
videos = list((out / "videos").glob("*.mp4"))
print(f"manifest_lines={len(rows)}")
print(f"reports={len(reports)}")
print(f"videos={len(videos)}")
print(f"natural_failures={sum(1 for r in rows if r.get('natural_failure_found'))}")
print(f"same_failure_passes={sum(1 for r in rows if r.get('same_failure_pass'))}")
print(f"causal_passes={sum(1 for r in rows if r.get('causal_pass'))}")
print(f"positive_windows={sum(int(r.get('positive_windows') or 0) for r in rows)}")
print(f"video_exists_rows={sum(1 for r in rows if r.get('video_exists'))}")
if (out / "summary.json").exists():
    print("summary_exists=1")
else:
    print("summary_exists=0")
PY

tail -n 8 "$OUTPUT_DIR/manifest.jsonl" 2>/dev/null || true
