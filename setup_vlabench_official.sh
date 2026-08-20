#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="/root/autodl-tmp/research"
REPO_DIR="${WORK_ROOT}/VLABench"
RRT_DIR="${WORK_ROOT}/rrt-algorithms"
LEROBOT_DIR="${WORK_ROOT}/lerobot-vlabench"
LEROBOT_COMMIT="6674e368249472c91382eb54bb8501c94c7f0c56"
ENV_PREFIX="/root/autodl-tmp/envs/vlabench"
LOG_DIR="/root/autodl-tmp/research/Embodied_Delta_Debugging"
LOG_FILE="${LOG_DIR}/vlabench_setup.log"
STATUS_FILE="${LOG_DIR}/vlabench_setup.status"
PACKAGE_ROOT="${REPO_DIR}/VLABench"
ASSET_DIR="${PACKAGE_ROOT}/assets"

mkdir -p "${WORK_ROOT}" "${LOG_DIR}" "/root/autodl-tmp/envs"

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
  mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/cache/pip /root/autodl-tmp/cache/huggingface
  export TMPDIR="/root/autodl-tmp/tmp"
  export TEMP="/root/autodl-tmp/tmp"
  export TMP="/root/autodl-tmp/tmp"
  export PIP_CACHE_DIR="/root/autodl-tmp/cache/pip"
  export HF_HOME="/root/autodl-tmp/cache/huggingface"
  export GIT_LFS_SKIP_SMUDGE=1
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
    sleep $((10 * n))
    n=$((n + 1))
  done
}

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "${dir}/.git" ]; then
    log "Using existing git repo ${dir}"
  else
    log "Cloning ${url} -> ${dir}"
    retry 3 git clone "${url}" "${dir}"
  fi
}

install_system_packages() {
  log "Checking/installing system simulation packages"
  no_proxy_env
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y libgl1 libegl1 libosmesa6 patchelf ffmpeg
}

create_env() {
  log "Creating/updating conda env at ${ENV_PREFIX}"
  no_proxy_env
  local conda_channels=(
    --override-channels
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  )
  if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    retry 3 conda create -y -p "${ENV_PREFIX}" python=3.10 pip "${conda_channels[@]}"
  else
    log "Conda env already exists"
  fi
  "${ENV_PREFIX}/bin/python" -m pip install -U pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
}

install_python_packages() {
  log "Installing Python packages"
  no_proxy_env
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
  export GIT_TERMINAL_PROMPT=0
  local filtered_requirements
  filtered_requirements="$(mktemp)"
  grep -v -E "huggingface/lerobot|motion-planning/rrt-algorithms" \
    "${REPO_DIR}/requirements.txt" > "${filtered_requirements}"
  "${ENV_PREFIX}/bin/python" -m pip install -r "${filtered_requirements}"
  rm -f "${filtered_requirements}"

  # The official VLABench pin uses a LeRobot commit whose metadata depends on
  # "pyav", but the installable PyPI project is named "av". Install LeRobot's
  # runtime deps explicitly with the corrected package name, then install that
  # official commit without asking pip to resolve the broken dependency name.
  "${ENV_PREFIX}/bin/python" -m pip install \
    "termcolor>=2.4.0" \
    "wandb>=0.16.3" \
    "imageio[ffmpeg]>=2.34.0" \
    "gdown>=5.1.0" \
    "einops>=0.8.0" \
    "pymunk>=6.6.0" \
    "zarr>=2.17.0" \
    "numba>=0.59.0" \
    "torch>=2.2.1" \
    "torchvision>=0.21.0" \
    "diffusers>=0.27.2" \
    "huggingface-hub[cli,hf-transfer]>=0.27.1" \
    "datasets>=2.19.0" \
    "av>=12.0.5" \
    "deepdiff>=7.0.1" \
    "flask>=3.0.3" \
    "jsonlines>=4.0.0"

  if ! "${ENV_PREFIX}/bin/python" - <<'PY'
import draccus
PY
  then
    "${ENV_PREFIX}/bin/python" -m pip install "draccus @ git+https://github.com/dlwh/draccus.git"
  fi

  if [ -d "${LEROBOT_DIR}/.git" ]; then
    log "Using existing LeRobot repo ${LEROBOT_DIR}"
  else
    retry 3 env GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
  fi
  if ! git -C "${LEROBOT_DIR}" cat-file -e "${LEROBOT_COMMIT}^{commit}" 2>/dev/null; then
    retry 3 git -C "${LEROBOT_DIR}" fetch origin "${LEROBOT_COMMIT}"
  fi
  env GIT_LFS_SKIP_SMUDGE=1 git -C "${LEROBOT_DIR}" reset --hard "${LEROBOT_COMMIT}"
  "${ENV_PREFIX}/bin/python" -m pip install --no-deps -e "${LEROBOT_DIR}"
  "${ENV_PREFIX}/bin/python" -m pip install "rtree==1.2.0" plotly
  "${ENV_PREFIX}/bin/python" -m pip install --no-deps -e "${RRT_DIR}" -e "${REPO_DIR}"
  "${ENV_PREFIX}/bin/python" -m pip install "numpy==1.25.0"
}

download_assets_from_hf() {
  log "Downloading VLABench assets from lerobot/vlabench-assets via hf-mirror"
  no_proxy_env
  export HF_ENDPOINT="https://hf-mirror.com"
  export HF_HUB_DISABLE_TELEMETRY=1
  mkdir -p "${ASSET_DIR}"
  retry 5 "${ENV_PREFIX}/bin/hf" download lerobot/vlabench-assets \
    --repo-type dataset \
    --local-dir "${ASSET_DIR}" \
    --max-workers 16
}

download_assets_official_fallback() {
  log "HF assets download failed; trying official Google Drive script as fallback"
  no_proxy_env
  export VLABENCH_ROOT="${PACKAGE_ROOT}"
  "${ENV_PREFIX}/bin/python" "${REPO_DIR}/scripts/download_assets.py"
}

init_submodules() {
  log "Initializing VLABench submodules"
  no_proxy_env
  git -C "${REPO_DIR}" submodule update --init --recursive
}

write_env_hook() {
  log "Writing conda activation environment variables"
  mkdir -p "${ENV_PREFIX}/etc/conda/activate.d"
  cat > "${ENV_PREFIX}/etc/conda/activate.d/vlabench_vars.sh" <<EOF
export VLABENCH_ROOT="${PACKAGE_ROOT}"
export MUJOCO_GL="egl"
EOF
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
print("python smoke ok")
print("VLABENCH_ROOT", root)
print("assets exists", assets.exists())
print("assets top entries", sorted(p.name for p in assets.iterdir())[:10] if assets.exists() else [])
print("mujoco", mujoco.__version__)
print("dm_control", getattr(dm_control, "__version__", "unknown"))
PY
}

main() {
  mark_status "running"
  log "=== VLABench official setup started ==="
  no_proxy_env
  log "Proxy variables cleared for this installer"
  install_system_packages
  clone_or_update "https://github.com/OpenMOSS/VLABench.git" "${REPO_DIR}"
  clone_or_update "https://github.com/motion-planning/rrt-algorithms.git" "${RRT_DIR}"
  create_env
  install_python_packages
  if ! download_assets_from_hf; then
    download_assets_official_fallback
  fi
  init_submodules
  write_env_hook
  smoke_test
  mark_status "done"
  log "=== VLABench official setup completed ==="
}

trap 'code=$?; mark_status "failed:${code}"; log "Installer failed with exit ${code}"; exit ${code}' ERR
main
