#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required before running this install step." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[install:20_torch] target conda env=${CONDA_ENV_NAME}"
echo "[install:20_torch] pip_tools=pip setuptools wheel uv"
python -m pip install --upgrade pip setuptools wheel uv

echo "[install:20_torch] cuda_index=${CUDA_INDEX_URL}"
echo "[install:20_torch] torch_packages=torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "${CUDA_INDEX_URL}"
