#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

# Step 1: install the local DreamerVLA package in editable mode.
install_log "target conda env=${CONDA_ENV_NAME} python=${PYTHON}"
install_log "repo_package=${DVLA_ROOT}"
uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}"

# Step 2: install the curated runtime dependency list.
install_log "requirements=${DVLA_ROOT}/requirements.txt"
uv pip install --python "${PYTHON}" -r "${DVLA_ROOT}/requirements.txt"

# Step 3: pin transformers for compatibility with the VLA backbones.
install_log "transformers=4.40.1"
uv pip install --python "${PYTHON}" transformers==4.40.1
