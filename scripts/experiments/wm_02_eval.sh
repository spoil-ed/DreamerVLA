#!/usr/bin/env bash
# Stage WM-02: evaluate a trained/warmup WM checkpoint for open/closed-loop drift.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[wm-eval] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-eval] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

: "${WM_CKPT:?set WM_CKPT=/path/to/wm_warmup.ckpt}"
: "${WM_CONFIG:?set WM_CONFIG=/path/to/cotrain/resolved_config.yaml}"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.diagnostics.eval_chunkwm_closeloop \
  --ckpt "${WM_CKPT}" \
  --config "${WM_CONFIG}" \
  --success-dir-raw "${SUCCESS_DIR_RAW:-${DVLA_DATA_ROOT}/processed_data/libero_goal_no_noops_t_256}" \
  --success-dir-hidden "${SUCCESS_DIR_HIDDEN:-${DVLA_DATA_ROOT}/processed_data/libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h1}" \
  --out "${WM_EVAL_OUT:-${DVLA_DATA_ROOT}/outputs/world_model_probe/wm_eval_summary.json}" \
  "$@"
