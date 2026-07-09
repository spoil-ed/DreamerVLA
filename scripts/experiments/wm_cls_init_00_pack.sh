#!/usr/bin/env bash
# Stage INIT-00: pack WM and classifier checkpoints into a cotrain init checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[wm-cls-init-pack] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-cls-init-pack] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

: "${WM_CKPT:?set WM_CKPT=/path/to/wm_warmup.ckpt}"
: "${CLASSIFIER_CKPT:?set CLASSIFIER_CKPT=/path/to/classifier.ckpt}"
INIT_CKPT="${INIT_CKPT:-${DVLA_DATA_ROOT}/outputs/world_model_probe/wm_cls_init.ckpt}"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.diagnostics.experiment_stage_checks pack-init \
  --wm-ckpt "${WM_CKPT}" \
  --classifier-ckpt "${CLASSIFIER_CKPT}" \
  --out "${INIT_CKPT}" \
  "$@"
