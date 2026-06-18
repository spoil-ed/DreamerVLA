#!/usr/bin/env bash
# E2E launcher: Ray cold-start collection -> offline-warmup online cotrain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray "$@"
