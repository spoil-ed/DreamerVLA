#!/usr/bin/env bash
# Stage COTRAIN-01: resume from warmup checkpoints and run online cotrain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[cotrain-run] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[cotrain-run] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

: "${RUN_ROOT:?set RUN_ROOT=/path/to/coldstart_warmup_cotrain/run}"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.launchers.coldstart_warmup_cotrain \
  mode="${MODE:-ray}" \
  task="${TASK:-goal}" \
  profile="${PROFILE:-release}" \
  cotrain_phase=online \
  skip_collect=true \
  run_root="${RUN_ROOT}" \
  python="${PYTHON_BIN}" \
  "$@"
