#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
source "$RUN_DIR/config.env"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [[ -n "${PYTORCH_DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--pytorch-device "$PYTORCH_DEVICE")
fi
if [[ -n "${PYTORCH_COMPILE_MODE:-}" ]]; then
  EXTRA_ARGS+=(--pytorch-compile-mode "$PYTORCH_COMPILE_MODE")
fi

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  /root/autodl-tmp/envs/libero38/bin/python -u "$PROJECT_ROOT/run_risk_critic_large_eval.py" \
  --output-dir "$OUTPUT_DIR" \
  --policy-port "$POLICY_PORT" \
  --launch-policy-server \
  --policy-server-kind "$POLICY_SERVER_KIND" \
  --policy-config "$POLICY_CONFIG" \
  --policy-dir "$POLICY_DIR" \
  --action-tokenizer-path "$ACTION_TOKENIZER_PATH" \
  --text-tokenizer-path "$TEXT_TOKENIZER_PATH" \
  --task-suite-name "$DATASET_NAME" \
  --task-ids "$TASK_IDS" \
  --init-state-ids "$INIT_STATE_IDS" \
  --seeds "$SEEDS" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --xla-mem-fraction "$XLA_MEM_FRACTION" \
  "${EXTRA_ARGS[@]}" \
  "$@"
