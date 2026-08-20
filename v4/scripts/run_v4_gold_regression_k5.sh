#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V4="$ROOT/v4"
CODE="$V4/code"
PY="/root/autodl-tmp/envs/libero38/bin/python"
LEROBOT_PY="/root/miniconda3/bin/python"
OUT="$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/risk_critic_ultra_video_causal_v4_gold_regression_k5_20260527"
POLICY_DIR="$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay"
ACTION_TOKENIZER="$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean"
TEXT_TOKENIZER="/root/autodl-tmp/research/VLA_SKILL/model/google/paligemma-3b-pt-224"
PORT="${PORT:-8060}"

mkdir -p "$OUT/reports" "$OUT/logs" "$OUT/videos"
ln -sfn "$OUT" "$V4/outputs/gold_regression_k5"

port_open() {
  "$PY" - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
}

if ! port_open; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    nohup "$LEROBOT_PY" "$CODE/serve_lerobot_pi0fast_policy.py" \
      --port "$PORT" \
      --policy-dir "$POLICY_DIR" \
      --device cuda \
      --compile-mode none \
      --action-tokenizer-path "$ACTION_TOKENIZER" \
      --text-tokenizer-path "$TEXT_TOKENIZER" \
      --local-files-only \
      > "$OUT/policy_server.log" 2>&1 &
  echo $! > "$OUT/policy_server.pid"
fi

for _ in $(seq 1 450); do
  if port_open; then
    break
  fi
  sleep 2
done
port_open

run_case() {
  local task="$1"
  local init="$2"
  local seed="$3"
  local prefix
  prefix="$(printf 'task%02d_init%02d_seed%02d' "$task" "$init" "$seed")"
  PYTHONPATH="$CODE" "$PY" "$CODE/pi05_natural_failure_probe.py" \
    --policy-host 127.0.0.1 \
    --policy-port "$PORT" \
    --policy-config pi0_fast_libero \
    --policy-checkpoint "$POLICY_DIR" \
    --task-suite-name libero_10 \
    --task-ids "$task" \
    --init-state-ids "$init" \
    --seed "$seed" \
    --replay-trials 5 \
    --search-replay-trials 1 \
    --confirm-replay-trials 5 \
    --repair-replay-trials 5 \
    --event-window 32 \
    --causal-context-before 48 \
    --causal-context-after 8 \
    --causal-max-units 18 \
    --causal-ablation-trials 5 \
    --causal-ablation-strategies hold,adjacent,gripper_correction \
    --repair-scheduling-mode topk_complete \
    --demo-repair-timeout-seconds 30 \
    --continuation recorded \
    --camera-size 512 \
    --gpu-id "${CUDA_VISIBLE_DEVICES:-0}" \
    --output "$OUT/reports/${prefix}_causal_v4.json" \
    --enable-visual-policy-mask \
    --record-video \
    --video-dir "$OUT/videos" \
    --video-prefix "$prefix" \
    --video-camera agentview_image \
    --video-fps 30 \
    --video-every-n 1 \
    --video-codec libx264 \
    --video-quality 10 \
    > "$OUT/logs/${prefix}.log" 2>&1 || true
}

run_case 1 15 7
run_case 8 46 17
run_case 9 1 27
run_case 8 21 17

echo "Wrote causal-v4 gold regression outputs to $OUT"
