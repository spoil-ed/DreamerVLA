#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

install_log "checking imports in conda env=${CONDA_ENV_NAME} python=${PYTHON}"
install_log "verifying imports and CUDA visibility"
"${PYTHON}" - <<'PY'
import torch
import h5py
import hydra
import omegaconf
import transformers
import libero

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("deps ok")
print("libero", libero.__path__)
PY
