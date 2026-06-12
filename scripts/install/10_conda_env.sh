#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Anaconda, then rerun this step." >&2
  exit 2
fi

echo "[install:10_conda_env] target conda env=${CONDA_ENV_NAME} python=${PYTHON_VERSION}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "[install:10_conda_env] conda env already exists: ${CONDA_ENV_NAME}"
fi
