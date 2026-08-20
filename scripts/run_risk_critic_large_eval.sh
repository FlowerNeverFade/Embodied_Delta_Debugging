#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/research/Embodied_Delta_Debugging"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/risk_critic_large_eval}"

cd "$PROJECT_ROOT"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
  /root/autodl-tmp/envs/libero38/bin/python \
  "$PROJECT_ROOT/run_risk_critic_large_eval.py" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
