#!/usr/bin/env bash
# Shared environment for DreamerVLA install step scripts.
set -euo pipefail

INSTALL_STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${INSTALL_STEP_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.1.post1}"
export FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
export INSTALL_OPTIONAL_THIRD_PARTY="${INSTALL_OPTIONAL_THIRD_PARTY:-0}"
export INSTALL_OPENSORA_THIRD_PARTY="${INSTALL_OPENSORA_THIRD_PARTY:-1}"
export INSTALL_OPENVLA_OFT_THIRD_PARTY="${INSTALL_OPENVLA_OFT_THIRD_PARTY:-1}"
export INSTALL_EGL_PROBE="${INSTALL_EGL_PROBE:-1}"
export INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-1}"

cd "${DVLA_ROOT}"

install_log() {
  printf '[install:%s] %s\n' "$(basename "$0")" "$*"
}

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required before running this install step." >&2
    exit 2
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
  export PYTHON="$(command -v python)"
}
