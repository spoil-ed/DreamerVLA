#!/usr/bin/env bash
# Eight-GPU Ray policy-only RL with explicit pretrained WM/CLS assignments.
# WORLD_MODEL_CKPT=/path/to/wm/run CLASSIFIER_CKPT=/path/to/classifier/run \
#   bash scripts/e2e_frozen_model_cotrain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
exec "${PYTHON_EXECUTABLE}" -m dreamervla.launchers.frozen_model_cotrain_ray "$@"
