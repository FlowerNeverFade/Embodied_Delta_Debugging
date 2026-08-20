#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"
mkdir -p "$OUTPUT_DIR"

: > "$OUTPUT_DIR/master.log"
setsid bash -c "cd '$RUN_DIR' && exec ./run_foreground.sh" \
  > "$OUTPUT_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR/run.pid"
echo "started pid=$(cat "$OUTPUT_DIR/run.pid")"
echo "log=$OUTPUT_DIR/master.log"
