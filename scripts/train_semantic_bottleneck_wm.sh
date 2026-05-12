#!/usr/bin/env bash
# Train the Dreamer-VLA writeup world model:
# frozen VLA z_sem -> Gaussian bottleneck z_phys -> Gaussian RSSM reward/continue.
#
# This is intentionally separate from train_wm.sh so the existing pixel/token/
# Rynn-backbone experiments keep their original defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/env_libero_goal.sh"

PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-semantic_bottleneck_wm_libero_goal}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29517}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
DATALOADER_MP_CONTEXT="${DATALOADER_MP_CONTEXT:-forkserver}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-${DREAMERVLA_UNIFIED_VLA_TAG}_zphys32_rssm_bs${BATCH_SIZE}_nw${NUM_WORKERS}_gpu${CUDA_VISIBLE_DEVICES//,/}}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/worldmodel/semantic_bottleneck_wm}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${RUN_TAG}_${TIMESTAMP}}"

COMMON_OVERRIDES=(
  "training.out_dir=${OUT_DIR}"
  "dataloader.batch_size=${BATCH_SIZE}"
  "dataloader.num_workers=${NUM_WORKERS}"
  "dataloader.pin_memory=${PIN_MEMORY}"
  "dataloader.persistent_workers=${PERSISTENT_WORKERS}"
  "dataloader.prefetch_factor=${PREFETCH_FACTOR}"
  "dataloader.multiprocessing_context=${DATALOADER_MP_CONTEXT}"
  "dataset.hidden_dir=${RYNN_HIDDEN_DIR}"
  "dataset.expected_model_path=${VLA_INIT_CKPT}"
  "dataset.expected_encoder_state_ckpt=${VLA_STATE_CKPT}"
  "dataset.expected_time_horizon=${ACTION_HORIZON}"
)

echo "Semantic bottleneck WM"
echo "Config:         ${CONFIG_NAME}"
echo "Run output dir: ${OUT_DIR}"
echo "GPUs:           ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NUM_GPUS})"
echo "Hidden sidecar: ${RYNN_HIDDEN_DIR}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN=1, not launching training. Command:\n'
  printf '%q ' "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    --module src.cli.train --config-name "${CONFIG_NAME}" "${COMMON_OVERRIDES[@]}" "$@"
  printf '\n'
  exit 0
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
