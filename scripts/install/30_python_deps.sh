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
echo "[install:30_python_deps] requirements=${DVLA_ROOT}/requirements.txt"
uv pip install -r "${DVLA_ROOT}/requirements.txt"

# transformers is intentionally NOT pinned here. OpenVLA-OFT needs moojink's
# transformers fork (bidirectional Llama attention; vanilla -> 0% garbage OFT
# actions), which 40_third_party.sh installs with --force-reinstall as the single
# authoritative transformers. Any transformers pulled transitively by the
# requirements above (peft/diffusers/tokenizers==0.19.1) is overridden there.

if [[ "${INSTALL_DEV_TOOLS}" == "1" ]]; then
  echo "[install:30_python_deps] dev_dependency_group=dev"
  uv pip install --group dev
else
  echo "[install:30_python_deps] dev_dependency_group=skipped"
fi
