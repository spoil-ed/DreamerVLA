#!/usr/bin/env bash
# Evaluate a trained full-dataset Chunk world model checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  DATA_ROOT_SOURCE="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  DATA_ROOT_SOURCE="DEFAULTED from DVLA_ROOT"
fi
echo "[world-model-training-eval] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[world-model-training-eval] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${DATA_ROOT_SOURCE})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
WORLD_MODEL_RUN_ROOT="${WORLD_MODEL_RUN_ROOT:-${RUN_ROOT:-}}"
if [[ -n "${WORLD_MODEL_RUN_ROOT}" ]]; then
  WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-${WORLD_MODEL_RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt}"
  WORLD_MODEL_CONFIG="${WORLD_MODEL_CONFIG:-${WORLD_MODEL_RUN_ROOT}/cotrain/resolved_config.yaml}"
fi
: "${WORLD_MODEL_CHECKPOINT:?set WORLD_MODEL_CHECKPOINT=/path/to/wm_warmup.ckpt or WORLD_MODEL_RUN_ROOT=/path/to/run}"
: "${WORLD_MODEL_CONFIG:?set WORLD_MODEL_CONFIG=/path/to/resolved_config.yaml or WORLD_MODEL_RUN_ROOT=/path/to/run}"

"${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.eval_chunkwm_closeloop \
  --ckpt "${WORLD_MODEL_CHECKPOINT}" \
  --config "${WORLD_MODEL_CONFIG}" \
  --success-dir-raw "${SUCCESS_DIR_RAW:-${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward}" \
  --success-dir-hidden "${SUCCESS_DIR_HIDDEN:-${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_input_token_embedding_vla_policy_h1}" \
  --out "${WORLD_MODEL_EVAL_OUTPUT:-${DVLA_DATA_ROOT}/outputs/world_model_probe/world_model_eval_summary.json}" \
  "$@"
