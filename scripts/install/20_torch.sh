#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

# Step 1: install pip tooling used by the later Torch and pip dependency steps.
install_log "target conda env=${CONDA_ENV_NAME} python=${PYTHON}"
install_log "pip_tools=pip setuptools wheel uv"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel uv

# Step 2: install the CUDA 12.4 PyTorch wheel set expected by DreamerVLA.
install_log "cuda_index=${CUDA_INDEX_URL}"
install_log "torch_packages=torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
uv pip install --python "${PYTHON}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url "${CUDA_INDEX_URL}"
