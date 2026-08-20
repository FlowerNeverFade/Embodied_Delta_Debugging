#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="/root/autodl-tmp/research"
REPO_DIR="${WORK_ROOT}/VLABench"
ENV_PREFIX="/root/autodl-tmp/envs/vlabench"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT"
LOG_FILE="${LOG_DIR}/vlabench_setup.log"
STATUS_FILE="${LOG_DIR}/vlabench_setup.status"
PACKAGE_ROOT="${REPO_DIR}/VLABench"
ASSET_DIR="${PACKAGE_ROOT}/assets"
DOWNLOADER="${LOG_DIR}/download_vlabench_assets_hf_mirror.py"

mkdir -p "${LOG_DIR}" "${ASSET_DIR}" /root/autodl-tmp/tmp /root/autodl-tmp/cache/huggingface
exec >> "${LOG_FILE}" 2>&1

log() {
  echo "[$(date '+%F %T %Z')] $*"
}

mark_status() {
  echo "$1" > "${STATUS_FILE}"
}

no_proxy_env() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="localhost,127.0.0.1,::1"
  export TMPDIR="/root/autodl-tmp/tmp"
  export TEMP="/root/autodl-tmp/tmp"
  export TMP="/root/autodl-tmp/tmp"
  export HF_HOME="/root/autodl-tmp/cache/huggingface"
  export HF_ENDPOINT="https://hf-mirror.com"
  export HF_HUB_DISABLE_TELEMETRY=1
  export GIT_LFS_SKIP_SMUDGE=1
  export PIP_CACHE_DIR="/root/autodl-tmp/cache/pip"
}

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    local code="$?"
    if [ "${n}" -ge "${attempts}" ]; then
      log "FAILED after ${n} attempts: $* (exit ${code})"
      return "${code}"
    fi
    log "Retry ${n}/${attempts} failed: $* (exit ${code}); sleeping..."
    sleep $((20 * n))
    n=$((n + 1))
  done
}

download_assets() {
  log "Downloading official VLABench assets from lerobot/vlabench-assets via hf-mirror"
  no_proxy_env
  "${ENV_PREFIX}/bin/python" "${DOWNLOADER}"
}

init_submodules() {
  log "Initializing VLABench submodules"
  no_proxy_env
  git -C "${REPO_DIR}" config submodule.third_party/openpi.url \
    "${OPENPI_GIT_URL:-https://gh-proxy.com/https://github.com/Shiduo-zh/openpi.git}"
  retry 5 git -C "${REPO_DIR}" submodule update --init third_party/openpi
  git -C "${REPO_DIR}/third_party/openpi" config submodule.third_party/aloha.url \
    "${ALOHA_GIT_URL:-https://gh-proxy.com/https://github.com/Physical-Intelligence/aloha.git}"
  git -C "${REPO_DIR}/third_party/openpi" config submodule.third_party/libero.url \
    "${LIBERO_GIT_URL:-https://gh-proxy.com/https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
  retry 5 git -C "${REPO_DIR}/third_party/openpi" submodule update --init --recursive
}

write_env_hook() {
  log "Writing conda activation environment variables"
  mkdir -p "${ENV_PREFIX}/etc/conda/activate.d"
  printf 'export VLABENCH_ROOT="%s"\nexport MUJOCO_GL="egl"\n' "${PACKAGE_ROOT}" \
    > "${ENV_PREFIX}/etc/conda/activate.d/vlabench_vars.sh"
}

smoke_test() {
  log "Running import/headless smoke test"
  no_proxy_env
  export VLABENCH_ROOT="${PACKAGE_ROOT}"
  export MUJOCO_GL="egl"
  "${ENV_PREFIX}/bin/python" - <<'PY'
import os
from pathlib import Path
import mujoco
import dm_control
import VLABench

root = Path(os.environ["VLABENCH_ROOT"])
assets = root / "assets"
files = sum(1 for path in assets.rglob("*") if path.is_file()) if assets.exists() else 0
print("python smoke ok")
print("VLABENCH_ROOT", root)
print("assets exists", assets.exists())
print("assets files", files)
print("assets top entries", sorted(p.name for p in assets.iterdir())[:10] if assets.exists() else [])
print("mujoco", mujoco.__version__)
print("dm_control", getattr(dm_control, "__version__", "unknown"))
PY
}

main() {
  mark_status "running-assets"
  log "=== VLABench no-proxy finish/resume started ==="
  no_proxy_env
  download_assets
  mark_status "running-submodules"
  init_submodules
  write_env_hook
  mark_status "running-smoke"
  smoke_test
  mark_status "done"
  log "=== VLABench no-proxy finish/resume completed ==="
}

trap 'code=$?; mark_status "failed:${code}"; log "Resume installer failed with exit ${code}"; exit "${code}"' ERR
main
