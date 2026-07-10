#!/usr/bin/env bash
# Train the configured Chunk-WM aggressively on the complete original LIBERO replay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[wm-full-dataset] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-full-dataset] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

# Stable 8xH100 runtime defaults. Every value remains environment-overridable.
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

PYTHON_BIN="${PYTHON:-python}"
NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
export RUN_ROOT="${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/wm_full_dataset/$(date +%Y%m%d_%H%M%S)}"

echo "[wm-full-dataset] GPUs      = ${CUDA_VISIBLE_DEVICES:-all}" >&2
echo "[wm-full-dataset] NGPU      = ${NGPU}" >&2
echo "[wm-full-dataset] EXPERIMENT= wm_full_dataset_train" >&2
echo "[wm-full-dataset] RUN_ROOT  = ${RUN_ROOT}" >&2

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NGPU}" \
  --master_port="${MASTER_PORT}" \
  -m dreamervla.train \
  experiment=wm_full_dataset_train \
  "$@"
