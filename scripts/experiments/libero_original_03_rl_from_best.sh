#!/usr/bin/env bash
# Stage LIBERO-ORIG-03: resume online RL from original-data WM+classifier warmup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[libero-original-rl] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[libero-original-rl] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

: "${RUN_ROOT:?set RUN_ROOT=/path/to/libero_original_best_wm_cls/run}"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.diagnostics.experiment_stage_checks libero-original-rl-run \
  --experiment "${ORIGINAL_COTRAIN_EXPERIMENT:-openvla_onetraj_libero_cotrain_noray}" \
  --task "${ORIGINAL_TASK:-openvla_onetraj_libero}" \
  --python "${PYTHON_BIN}" \
  --ngpu "${NGPU:-1}" \
  --master-port "${MASTER_PORT:-29500}" \
  --run-root "${RUN_ROOT}" \
  --total-env-steps "${RL_TOTAL_ENV_STEPS:-200000}" \
  --task-ids "${ORIGINAL_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}" \
  --render-backend "${RL_RENDER_BACKEND:-osmesa}" \
  "$@"
