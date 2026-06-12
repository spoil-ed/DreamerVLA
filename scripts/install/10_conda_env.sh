#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Anaconda, then rerun this step." >&2
  exit 2
fi

install_log "target conda env=${CONDA_ENV_NAME} python=${PYTHON_VERSION}"
install_log "ensuring conda env ${CONDA_ENV_NAME} with Python ${PYTHON_VERSION}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  install_log "conda env already exists: ${CONDA_ENV_NAME}"
fi
