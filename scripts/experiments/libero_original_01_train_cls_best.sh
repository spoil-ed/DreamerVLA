#!/usr/bin/env bash
# Stage LIBERO-ORIG-01: train a high-budget standalone success classifier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[libero-original-cls-best] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[libero-original-cls-best] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.diagnostics.experiment_stage_checks libero-original-cls-run \
  --experiment "${CLS_EXPERIMENT:-wmpo_token_classifier_openvla_onetraj_libero_goal_h1}" \
  --task "${ORIGINAL_TASK:-openvla_onetraj_libero}" \
  --python "${PYTHON_BIN}" \
  --epochs "${CLS_EPOCHS:-32}" \
  --batch-size "${CLS_BATCH_SIZE:-16}" \
  --val-batch-size "${CLS_VAL_BATCH_SIZE:-64}" \
  --lr "${CLS_LR:-3.0e-5}" \
  --eval-every "${CLS_EVAL_EVERY:-100}" \
  --ckpt-every "${CLS_CKPT_EVERY:-100}" \
  "$@"
