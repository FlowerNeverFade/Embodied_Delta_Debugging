#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
PID_FILE="$OUTPUT_DIR/run.pid"
echo "run_dir=$RUN_DIR"
echo "output_dir=$OUTPUT_DIR"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,ppid,stat,sid,etime,cmd || true
else
  echo "no pid file"
fi
echo "reports=$(find "$OUTPUT_DIR/reports" -type f -name '*causal_v1.json' 2>/dev/null | wc -l)"
echo "manifest_lines=$(wc -l < "$OUTPUT_DIR/manifest.jsonl" 2>/dev/null || echo 0)"
tail -n 8 "$OUTPUT_DIR/manifest.jsonl" 2>/dev/null || true
