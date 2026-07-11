#!/usr/bin/env bash
# Check data and train the WMPO token success classifier with Hydra.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  DATA_ROOT_SOURCE="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  DATA_ROOT_SOURCE="DEFAULTED from DVLA_ROOT"
fi
echo "[classifier-training] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[classifier-training] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${DATA_ROOT_SOURCE})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
GPU_COUNT="${GPU_COUNT:-${NGPU:-8}}"
MASTER_PORT="${MASTER_PORT:-29501}"
CLASSIFIER_EXPERIMENT="${CLASSIFIER_EXPERIMENT:-wmpo_token_classifier_openvla_onetraj_libero_goal_h1}"
CLASSIFIER_RESUME="${CLASSIFIER_RESUME:-${RESUME:-false}}"
if [[ "${CLASSIFIER_RESUME}" == "true" && -z "${CLASSIFIER_RUN_ROOT:-}" && -z "${CLASSIFIER_RESUME_DIR:-}" ]]; then
  echo "[classifier-training] CLASSIFIER_RESUME=true requires CLASSIFIER_RUN_ROOT or CLASSIFIER_RESUME_DIR" >&2
  exit 2
fi
if [[ -n "${CLASSIFIER_RESUME_DIR:-}" ]]; then
  export CLASSIFIER_RUN_ROOT="${CLASSIFIER_RUN_ROOT:-${CLASSIFIER_RESUME_DIR}}"
else
  export CLASSIFIER_RUN_ROOT="${CLASSIFIER_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/classifier/wmpo_token_classifier_openvla_onetraj_libero_goal_h1/$(date +%Y%m%d_%H%M%S)}"
fi
CLASSIFIER_RESUME_DIR="${CLASSIFIER_RESUME_DIR:-${CLASSIFIER_RUN_ROOT}}"
CLASSIFIER_CHECKPOINT_EVERY="${CLASSIFIER_CHECKPOINT_EVERY:-250}"
if [[ "${CLASSIFIER_RESUME}" == "true" ]]; then
  if [[ -d "${CLASSIFIER_RESUME_DIR}" ]]; then
    if [[ ! -f "${CLASSIFIER_RESUME_DIR}/checkpoints/latest.ckpt" \
      && ! -f "${CLASSIFIER_RESUME_DIR}/ckpt/latest.ckpt" \
      && ! -f "${CLASSIFIER_RESUME_DIR}/latest.ckpt" ]]; then
      echo "[classifier-training] no latest classifier checkpoint under CLASSIFIER_RESUME_DIR=${CLASSIFIER_RESUME_DIR}" >&2
      exit 2
    fi
  else
    echo "[classifier-training] classifier resume run directory not found: ${CLASSIFIER_RESUME_DIR}" >&2
    exit 2
  fi
fi

echo "[classifier-training] GPUS       = ${CUDA_VISIBLE_DEVICES:-all}" >&2
echo "[classifier-training] GPU_COUNT  = ${GPU_COUNT}" >&2
echo "[classifier-training] EXPERIMENT = ${CLASSIFIER_EXPERIMENT}" >&2
echo "[classifier-training] RUN_ROOT   = ${CLASSIFIER_RUN_ROOT}" >&2
echo "[classifier-training] RESUME     = ${CLASSIFIER_RESUME}" >&2
echo "[classifier-training] CHECKPOINT_EVERY = ${CLASSIFIER_CHECKPOINT_EVERY}" >&2

"${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.experiment_stage_checks cls-check \
  --experiment "${CLASSIFIER_EXPERIMENT}" \
  "$@"

"${PYTHON_EXECUTABLE}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPU_COUNT}" \
  --master_port="${MASTER_PORT}" \
  -m dreamervla.train \
  "experiment=${CLASSIFIER_EXPERIMENT}" \
  training.out_dir="${CLASSIFIER_RUN_ROOT}" \
  training.resume="${CLASSIFIER_RESUME}" \
  ++training.resume_dir="${CLASSIFIER_RESUME_DIR}" \
  training.ckpt_every="${CLASSIFIER_CHECKPOINT_EVERY}" \
  ++training.distributed_strategy=ddp \
  "$@"
