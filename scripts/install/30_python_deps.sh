#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-1}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required before running this install step." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[install:30_python_deps] target conda env=${CONDA_ENV_NAME}"
echo "[install:30_python_deps] repo_package=${DVLA_ROOT}"
uv pip install -e "${DVLA_ROOT}"

echo "[install:30_python_deps] requirements=${DVLA_ROOT}/requirements.txt"
uv pip install -r "${DVLA_ROOT}/requirements.txt"

echo "[install:30_python_deps] transformers=4.40.1"
uv pip install transformers==4.40.1

if [[ "${INSTALL_DEV_TOOLS}" == "1" ]]; then
  echo "[install:30_python_deps] dev_dependency_group=dev"
  uv pip install --group dev
else
  echo "[install:30_python_deps] dev_dependency_group=skipped"
fi
