#!/usr/bin/env bash
# One-click official-data Chunk world-model training through Hydra.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  DATA_ROOT_SOURCE="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  DATA_ROOT_SOURCE="DEFAULTED from DVLA_ROOT"
fi
echo "[world-model-training] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[world-model-training] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${DATA_ROOT_SOURCE})" >&2
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
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_MODEL_EXPERIMENT="${WORLD_MODEL_EXPERIMENT:-wm_official_upper_bound}"
WORLD_MODEL_RESUME="${WORLD_MODEL_RESUME:-${RESUME:-false}}"
if [[ "${WORLD_MODEL_RESUME}" == "true" && -z "${WORLD_MODEL_RUN_ROOT:-}" && -z "${RUN_ROOT:-}" ]]; then
  echo "[world-model-training] WORLD_MODEL_RESUME=true requires WORLD_MODEL_RUN_ROOT or RUN_ROOT" >&2
  exit 2
fi
export WORLD_MODEL_RUN_ROOT="${WORLD_MODEL_RUN_ROOT:-${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/world_model/$(date +%Y%m%d_%H%M%S)}}"
export RUN_ROOT="${WORLD_MODEL_RUN_ROOT}"
WORLD_MODEL_CHECKPOINT_EVERY="${WORLD_MODEL_CHECKPOINT_EVERY:-500}"
WORLD_MODEL_TOPK_K="${WORLD_MODEL_TOPK_K:-3}"
if [[ "${WORLD_MODEL_RESUME}" == "true" ]]; then
  WORLD_MODEL_PROGRESS_GLOB="${WORLD_MODEL_RUN_ROOT}/ckpt/warmup_progress/wm_step_*.ckpt"
  if [[ ! -f "${WORLD_MODEL_RUN_ROOT}/ckpt/wm_warmup.ckpt" \
    && ! -d "${WORLD_MODEL_RUN_ROOT}/ckpt/wm_warmup_hf" \
    && -z "$(compgen -G "${WORLD_MODEL_PROGRESS_GLOB}")" ]]; then
    echo "[world-model-training] no world model warmup checkpoint/progress under ${WORLD_MODEL_RUN_ROOT}/ckpt" >&2
    exit 2
  fi
fi

echo "[world-model-training] GPUS       = ${CUDA_VISIBLE_DEVICES:-all}" >&2
echo "[world-model-training] GPU_COUNT  = ${GPU_COUNT}" >&2
echo "[world-model-training] EXPERIMENT = ${WORLD_MODEL_EXPERIMENT}" >&2
echo "[world-model-training] RUN_ROOT   = ${WORLD_MODEL_RUN_ROOT}" >&2
echo "[world-model-training] RESUME     = ${WORLD_MODEL_RESUME}" >&2
echo "[world-model-training] CHECKPOINT_EVERY = ${WORLD_MODEL_CHECKPOINT_EVERY}" >&2
echo "[world-model-training] TOPK_K     = ${WORLD_MODEL_TOPK_K}" >&2

"${PYTHON_EXECUTABLE}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPU_COUNT}" \
  --master_port="${MASTER_PORT}" \
  -m dreamervla.train \
  experiment=${WORLD_MODEL_EXPERIMENT} \
  training.out_dir="${WORLD_MODEL_RUN_ROOT}" \
  training.resume="${WORLD_MODEL_RESUME}" \
  training.warmup_checkpoint_every="${WORLD_MODEL_CHECKPOINT_EVERY}" \
  training.warmup_topk_k="${WORLD_MODEL_TOPK_K}" \
  "$@"
