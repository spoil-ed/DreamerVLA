#!/usr/bin/env bash
# Stage WM-01: run WM-only warmup from collected rollout replay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[wm-train] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-train] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.launchers.coldstart_warmup_cotrain \
  mode="${MODE:-ray}" \
  task="${TASK:-goal}" \
  profile="${PROFILE:-release}" \
  cotrain_phase=warmup \
  skip_collect=true \
  warmup.classifier_steps=0 \
  warmup.total_env_steps=0 \
  python="${PYTHON_BIN}" \
  "$@"
