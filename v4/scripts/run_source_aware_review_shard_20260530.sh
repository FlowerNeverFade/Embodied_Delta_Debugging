#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 GPU PORT OUTPUT_DIR REPORT_JSON [REPORT_JSON ...]" >&2
  exit 2
fi

GPU="$1"
PORT="$2"
OUT="$3"
shift 3

ROOT="/data2/yanghaoyun/research/Embodied_Delta_Debugging"
CODE="$ROOT/v4/code"
PY="/data2/yanghaoyun/envs/libero38/bin/python"
NULL_REPORT_DIR="$OUT/nonexistent_report_dir"
POLICY_DIR="$ROOT/model_datasets/pi0fast-libero-libero_10/policy_overlay"
ACTION_TOKENIZER="$ROOT/model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean"
TEXT_TOKENIZER="/data2/yanghaoyun/research/VLA_SKILL/model/google/paligemma-3b-pt-224"
DEMO_ROOT="/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"

export EDD_PROJECT_ROOT="$ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero_config"
export OPENPI_PYTHON="$PY"
export LEROBOT_PYTHON="/data2/yanghaoyun/miniconda3/bin/python"
export OPENPI_CLIENT_SRC="/data2/yanghaoyun/research/openpi/packages/openpi-client/src"
export PYTHONPATH="$CODE:$OPENPI_CLIENT_SRC:/data2/yanghaoyun/research/LIBERO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"
echo "$$" > "$OUT/run.pid"
printf "%s\n" "$@" > "$OUT/report_paths.txt"
echo "$(date -Is) shard starting gpu=$GPU port=$PORT reports=$#" | tee "$OUT/shard.log"

args=()
for report in "$@"; do
  if [[ ! -f "$report" ]]; then
    echo "missing report: $report" | tee -a "$OUT/shard.log" >&2
    exit 2
  fi
  args+=(--report-path "$report")
done

echo "$(date -Is) invoking slice_review_export.py" | tee -a "$OUT/shard.log"
set +e
CUDA_VISIBLE_DEVICES="$GPU" \
MUJOCO_EGL_DEVICE_ID="$GPU" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  "$PY" "$CODE/slice_review_export.py" \
    "${args[@]}" \
    --report-dir "$NULL_REPORT_DIR" \
    --output-dir "$OUT/review" \
    --include-necessity-only \
    --replay-trials-to-record 1 \
    --policy-repair-review-mode "${POLICY_REPAIR_REVIEW_MODE:-recorded_error}" \
    --review-top-k-sets 1 \
    --task-suite-name libero_10 \
    --policy-port "$PORT" \
    --launch-policy-server \
    --cuda-visible-devices "$GPU" \
    --policy-dir "$POLICY_DIR" \
    --action-tokenizer-path "$ACTION_TOKENIZER" \
    --text-tokenizer-path "$TEXT_TOKENIZER" \
    --demo-dataset-root "$DEMO_ROOT" \
    --camera-size 512 \
    --resize-size 224 \
    --video-fps 30 \
    --video-codec libx264 \
    --video-quality 10 \
    > "$OUT/review_export.log" 2>&1
status=$?
set -e
echo "$(date -Is) slice_review_export.py exit_status=$status" | tee -a "$OUT/shard.log"
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "$(date -Is) shard finished gpu=$GPU" | tee -a "$OUT/shard.log"
