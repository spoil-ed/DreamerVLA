#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

clone_repo() {
  local url="$1"
  local dest="$2"
  local ref="$3"
  install_log "ensuring $(basename "${dest}") ref=${ref} -> ${dest}"
  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --all --tags
  git -C "${dest}" checkout "${ref}"
}

install_log "special_packages=flash-attn egl_probe optional_apex optional_TensorNVMe"
install_log "flash_attn=${INSTALL_FLASH_ATTN} version=${FLASH_ATTN_VERSION}"
install_log "egl_probe=${INSTALL_EGL_PROBE}"
install_log "optional_third_party=${INSTALL_OPTIONAL_THIRD_PARTY}"
install_log "wheel_cache=${DVLA_DATA_ROOT}/wheels"

# Step 1: flash-attn is installed from a wheel because it is slow and fragile to build locally.
if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  mkdir -p "${DVLA_DATA_ROOT}/wheels"
  FLASH_ATTN_WHEEL="${DVLA_DATA_ROOT}/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
  install_log "installing flash-attn wheel=${FLASH_ATTN_WHEEL}"
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    curl -L "${FLASH_ATTN_WHEEL_URL}" -o "${FLASH_ATTN_WHEEL}"
  fi
  uv pip install --python "${PYTHON}" "${FLASH_ATTN_WHEEL}"
else
  install_log "skip flash-attn because INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"
fi

# Step 2: egl_probe validates EGL rendering for LIBERO/robosuite rollouts.
if [[ "${INSTALL_EGL_PROBE}" == "1" ]]; then
  clone_repo https://github.com/StanfordVL/egl_probe.git "${DVLA_ROOT}/third_party/egl_probe" 3ddf90d
  if [[ -f "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" ]]; then
    sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
      "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" || true
  fi
  uv pip install --python "${PYTHON}" --no-build-isolation "${DVLA_ROOT}/third_party/egl_probe"
else
  install_log "skip egl_probe because INSTALL_EGL_PROBE=${INSTALL_EGL_PROBE}"
fi

# Step 3: optional GPU acceleration packages are useful on some hosts but not required.
if [[ "${INSTALL_OPTIONAL_THIRD_PARTY}" == "1" ]]; then
  clone_repo https://github.com/NVIDIA/apex.git "${DVLA_ROOT}/third_party/apex" 5daec2a
  clone_repo https://github.com/hpcaitech/TensorNVMe.git "${DVLA_ROOT}/third_party/TensorNVMe" 6403388
  uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}/third_party/TensorNVMe" || true
  uv pip install --python "${PYTHON}" -v --no-build-isolation "${DVLA_ROOT}/third_party/apex" || true
else
  install_log "skip optional apex/TensorNVMe because INSTALL_OPTIONAL_THIRD_PARTY=${INSTALL_OPTIONAL_THIRD_PARTY}"
fi
