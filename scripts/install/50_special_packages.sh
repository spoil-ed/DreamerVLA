#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.1.post1}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_EGL_PROBE="${INSTALL_EGL_PROBE:-1}"
INSTALL_OPTIONAL_THIRD_PARTY="${INSTALL_OPTIONAL_THIRD_PARTY:-0}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required before running this install step." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[install:50_special_packages] special_packages=flash-attn egl_probe optional_apex optional_TensorNVMe"
echo "[install:50_special_packages] flash_attn=${INSTALL_FLASH_ATTN} version=${FLASH_ATTN_VERSION}"
echo "[install:50_special_packages] egl_probe=${INSTALL_EGL_PROBE}"
echo "[install:50_special_packages] optional_third_party=${INSTALL_OPTIONAL_THIRD_PARTY}"
echo "[install:50_special_packages] wheel_cache=${DVLA_DATA_ROOT}/wheels"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  mkdir -p "${DVLA_DATA_ROOT}/wheels"
  FLASH_ATTN_WHEEL="${DVLA_DATA_ROOT}/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    curl -L "${FLASH_ATTN_WHEEL_URL}" -o "${FLASH_ATTN_WHEEL}"
  fi
  uv pip install "${FLASH_ATTN_WHEEL}"
fi

if [[ "${INSTALL_EGL_PROBE}" == "1" ]]; then
  if [[ ! -d "${DVLA_ROOT}/third_party/egl_probe/.git" ]]; then git clone https://github.com/StanfordVL/egl_probe.git "${DVLA_ROOT}/third_party/egl_probe"; fi
  git -C "${DVLA_ROOT}/third_party/egl_probe" fetch --all --tags
  git -C "${DVLA_ROOT}/third_party/egl_probe" checkout 3ddf90d
  if [[ -f "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" ]]; then
    sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
      "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" || true
  fi
  uv pip install --no-build-isolation "${DVLA_ROOT}/third_party/egl_probe"
fi

if [[ "${INSTALL_OPTIONAL_THIRD_PARTY}" == "1" ]]; then
  if [[ ! -d "${DVLA_ROOT}/third_party/apex/.git" ]]; then git clone https://github.com/NVIDIA/apex.git "${DVLA_ROOT}/third_party/apex"; fi
  git -C "${DVLA_ROOT}/third_party/apex" fetch --all --tags
  git -C "${DVLA_ROOT}/third_party/apex" checkout 5daec2a
  if [[ ! -d "${DVLA_ROOT}/third_party/TensorNVMe/.git" ]]; then git clone https://github.com/hpcaitech/TensorNVMe.git "${DVLA_ROOT}/third_party/TensorNVMe"; fi
  git -C "${DVLA_ROOT}/third_party/TensorNVMe" fetch --all --tags
  git -C "${DVLA_ROOT}/third_party/TensorNVMe" checkout 6403388
  uv pip install -e "${DVLA_ROOT}/third_party/TensorNVMe" || true
  uv pip install -v --no-build-isolation "${DVLA_ROOT}/third_party/apex" || true
fi
