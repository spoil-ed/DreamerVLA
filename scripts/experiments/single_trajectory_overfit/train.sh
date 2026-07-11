#!/usr/bin/env bash
# Train the configured world model on one LIBERO trajectory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  DATA_ROOT_SOURCE="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  DATA_ROOT_SOURCE="DEFAULTED from DVLA_ROOT"
fi
echo "[single-trajectory-overfit] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[single-trajectory-overfit] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${DATA_ROOT_SOURCE})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
SINGLE_TRAJECTORY_TASK="${SINGLE_TRAJECTORY_TASK:-openvla_onetraj_libero}"
export SINGLE_TRAJECTORY_RUN_ROOT="${SINGLE_TRAJECTORY_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/single_trajectory_overfit/$(date +%Y%m%d_%H%M%S)}"

echo "[single-trajectory-overfit] TASK     = ${SINGLE_TRAJECTORY_TASK}" >&2
echo "[single-trajectory-overfit] RUN_ROOT = ${SINGLE_TRAJECTORY_RUN_ROOT}" >&2

"${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.wm_single_trajectory_overfit \
  --task "${SINGLE_TRAJECTORY_TASK}" \
  --out-dir "${SINGLE_TRAJECTORY_RUN_ROOT}" \
  "$@"

"${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.wm_single_trajectory_overfit \
  --run \
  --task "${SINGLE_TRAJECTORY_TASK}" \
  --out-dir "${SINGLE_TRAJECTORY_RUN_ROOT}" \
  "$@"
