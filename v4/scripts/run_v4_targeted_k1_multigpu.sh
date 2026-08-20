#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

ROOT="/root/autodl-tmp/research/Embodied_Delta_Debugging"
V4="$ROOT/v4"
CODE="$V4/code"
PY="/root/autodl-tmp/envs/libero38/bin/python"
OUT="$ROOT/model_datasets/pi0fast-libero-libero_10/outputs/risk_critic_ultra_video_causal_v4_targeted_k1_20260527"
POLICY_DIR="$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay"
ACTION_TOKENIZER="$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean"
TEXT_TOKENIZER="/root/autodl-tmp/research/VLA_SKILL/model/google/paligemma-3b-pt-224"

mkdir -p "$OUT"
ln -sfn "$OUT" "$V4/outputs/targeted_k1"

launch_shard() {
  local gpu="$1"
  local port="$2"
  local init_ids="$3"
  local shard_dir="$OUT/gpu${gpu}"
  mkdir -p "$shard_dir/reports" "$shard_dir/logs" "$shard_dir/videos"
  PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    nohup "$PY" -u "$CODE/run_risk_critic_large_eval.py" \
      --output-dir "$shard_dir" \
      --report-dir "$shard_dir/reports" \
      --log-dir "$shard_dir/logs" \
      --policy-port "$port" \
      --launch-policy-server \
      --policy-server-kind lerobot_pi0fast \
      --policy-config pi0_fast_libero \
      --policy-dir "$POLICY_DIR" \
      --action-tokenizer-path "$ACTION_TOKENIZER" \
      --text-tokenizer-path "$TEXT_TOKENIZER" \
      --pytorch-device cuda \
      --pytorch-compile-mode none \
      --task-suite-name libero_10 \
      --task-ids 1,4,8,9 \
      --init-state-ids "$init_ids" \
      --seeds 7,17,27 \
      --max-cases 100000 \
      --positive-target 0 \
      --min-cases-before-positive-stop 0 \
      --shuffle-cases \
      --case-order-seed 20260527 \
      --cuda-visible-devices "$gpu" \
      --xla-mem-fraction 0.55 \
      --search-replay-trials 1 \
      --confirm-replay-trials 1 \
      --repair-replay-trials 1 \
      --replay-trials 1 \
      --continuation recorded \
      --event-window 32 \
      --causal-context-before 48 \
      --causal-context-after 8 \
      --causal-max-units 18 \
      --causal-ablation-trials 1 \
      --causal-ablation-strategies hold \
      --repair-scheduling-mode pass_hunt \
      --demo-repair-timeout-seconds 30 \
      --case-timeout-seconds 1200 \
      --train-steps 1 \
      --record-video \
      --require-video \
      --video-dir "$shard_dir/videos" \
      --video-camera agentview_image \
      --video-fps 30 \
      --video-every-n 1 \
      --video-codec libx264 \
      --video-quality 10 \
      --camera-size 512 \
      --skip-source-repair-if-policy-pass \
      --stop-after-first-repair-valid-core \
      --defer-source-repair \
      --enable-visual-policy-mask \
      > "$shard_dir/master.log" 2>&1 &
  echo $! > "$shard_dir/run.pid"
}

launch_shard 0 8060 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
launch_shard 1 8061 "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33"
launch_shard 2 8062 "34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"

echo "Started causal-v4 targeted K=1 shards."
echo "Logs:"
echo "  tail -f $OUT/gpu0/master.log"
echo "  tail -f $OUT/gpu1/master.log"
echo "  tail -f $OUT/gpu2/master.log"
