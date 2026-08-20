#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.env"

if [[ -f "$OUTPUT_DIR/run.pid" ]]; then
  kill "$(cat "$OUTPUT_DIR/run.pid")" 2>/dev/null || true
fi
if [[ -f "$OUTPUT_DIR/policy_server.pid" ]]; then
  kill "$(cat "$OUTPUT_DIR/policy_server.pid")" 2>/dev/null || true
fi
pkill -f "[s]erve_policy.py --port $POLICY_PORT" 2>/dev/null || true
pkill -f "[s]erve_lerobot_pi0fast_policy.py --port $POLICY_PORT" 2>/dev/null || true
pkill -f "[r]un_risk_critic_large_eval.py .*--policy-port $POLICY_PORT" 2>/dev/null || true
pkill -f "[p]i05_natural_failure_probe.py .*--policy-port $POLICY_PORT" 2>/dev/null || true
echo "stopped $MODEL_NAME-$DATASET_NAME on port $POLICY_PORT"
