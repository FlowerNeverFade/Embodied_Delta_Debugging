#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

OUTPUT_DIR="${ULTRA_OUTPUT_DIR:-$RUN_DIR/outputs/risk_critic_ultra_video_causal_v3_targeted_k1_20260527}"
PID_FILE="$OUTPUT_DIR/run.pid"
POLICY_PID_FILE="$OUTPUT_DIR/policy_server.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  PGID="$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$PGID" ]]; then
    kill "-$PGID" 2>/dev/null || true
  else
    kill "$PID" 2>/dev/null || true
  fi
fi

if [[ -f "$POLICY_PID_FILE" ]]; then
  kill "$(cat "$POLICY_PID_FILE")" 2>/dev/null || true
fi

pkill -f "[r]un_risk_critic_large_eval.py .*--policy-port $POLICY_PORT" 2>/dev/null || true
pkill -f "[p]i05_natural_failure_probe.py .*--policy-port $POLICY_PORT" 2>/dev/null || true
pkill -f "[s]erve_lerobot_pi0fast_policy.py --port $POLICY_PORT" 2>/dev/null || true
pkill -f "[s]erve_policy.py --port $POLICY_PORT" 2>/dev/null || true

sleep 2
echo "paused output_dir=$OUTPUT_DIR"
echo "reports=$(find "$OUTPUT_DIR/reports" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
echo "videos=$(find "$OUTPUT_DIR/videos" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
