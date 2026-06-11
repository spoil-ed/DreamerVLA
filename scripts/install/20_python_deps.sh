#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

install_log "installing pip, uv, torch, repo package, and requirements"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel uv
uv pip install --python "${PYTHON}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url "${CUDA_INDEX_URL}"
uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}"
uv pip install --python "${PYTHON}" -r "${DVLA_ROOT}/requirements.txt"
uv pip install --python "${PYTHON}" transformers==4.40.1

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  install_log "installing flash-attn wheel"
  mkdir -p "${DVLA_DATA_ROOT}/wheels"
  FLASH_ATTN_WHEEL="${DVLA_DATA_ROOT}/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    curl -L "${FLASH_ATTN_WHEEL_URL}" -o "${FLASH_ATTN_WHEEL}"
  fi
  uv pip install --python "${PYTHON}" "${FLASH_ATTN_WHEEL}"
else
  install_log "skip flash-attn because INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"
fi
