#!/usr/bin/env bash
# Stage COTRAIN-02: run LIBERO eval for a trained cotrain/Dreamer checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[cotrain-eval] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[cotrain-eval] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

: "${EVAL_CKPT:?set EVAL_CKPT=/path/to/dreamer_or_vla_checkpoint}"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.launchers.train --config-name eval_libero_vla \
  task="${TASK:-goal}" \
  eval.ckpt_path="${EVAL_CKPT}" \
  eval.ckpt_kind="${EVAL_CKPT_KIND:-dreamer}" \
  "$@"
