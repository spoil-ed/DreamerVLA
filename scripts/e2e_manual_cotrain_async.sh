#!/usr/bin/env bash
# E2E launcher: current manual async OpenVLA-OFT cotrain route.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
if [[ -z "${DVLA_DATA_ROOT:-}" ]]; then
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
fi
echo "[manual-cotrain] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[manual-cotrain] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT}" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

python -m dreamervla.launchers.manual_cotrain_async "$@"
