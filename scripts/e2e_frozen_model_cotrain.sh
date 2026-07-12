#!/usr/bin/env bash
# Policy-only RL with positional pretrained WM/CLS checkpoints.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <world-model.ckpt> <classifier.ckpt> [hydra overrides...]" >&2
  exit 2
fi

WORLD_MODEL_CKPT="${1:-}"
CLASSIFIER_CKPT="${2:-}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
COTRAIN_RESUME="${COTRAIN_RESUME:-${RESUME:-false}}"

if [[ -z "${WORLD_MODEL_CKPT}" || ! -f "${WORLD_MODEL_CKPT}" ]]; then
  echo "[frozen-model-cotrain] WORLD_MODEL_CKPT must be an existing checkpoint" >&2
  exit 2
fi
if [[ -z "${CLASSIFIER_CKPT}" || ! -f "${CLASSIFIER_CKPT}" ]]; then
  echo "[frozen-model-cotrain] CLASSIFIER_CKPT must be an existing checkpoint" >&2
  exit 2
fi
if [[ "${COTRAIN_RESUME}" == "true" && -z "${COTRAIN_RUN_ROOT:-}" ]]; then
  echo "[frozen-model-cotrain] COTRAIN_RESUME=true requires COTRAIN_RUN_ROOT" >&2
  exit 2
fi

export COTRAIN_RUN_ROOT="${COTRAIN_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/frozen_cotrain/$(date +%Y%m%d_%H%M%S)}"

echo "[frozen-model-cotrain] WM_CKPT       = ${WORLD_MODEL_CKPT}" >&2
echo "[frozen-model-cotrain] CLS_CKPT      = ${CLASSIFIER_CKPT}" >&2
echo "[frozen-model-cotrain] RUN_ROOT      = ${COTRAIN_RUN_ROOT}" >&2
echo "[frozen-model-cotrain] RESUME        = ${COTRAIN_RESUME}" >&2

"${PYTHON_EXECUTABLE}" -m dreamervla.train \
  experiment=dreamervla_frozen_models_rl \
  task=openvla_onetraj_libero \
  training.out_dir="${COTRAIN_RUN_ROOT}" \
  training.resume="${COTRAIN_RESUME}" \
  training.resume_dir="${COTRAIN_RUN_ROOT}" \
  init.world_model_state_ckpt="${WORLD_MODEL_CKPT}" \
  init.classifier_state_ckpt="${CLASSIFIER_CKPT}" \
  "$@"
