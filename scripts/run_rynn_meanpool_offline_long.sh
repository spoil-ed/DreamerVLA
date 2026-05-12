#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="/home/user01/miniconda3/envs/dreamervla/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_rynn_pixel_precomputed_actor}"
export RUN_TAG="${RUN_TAG:-offline_rynn_meanpool_dreamer_actor_long_bs${BATCH_SIZE:-24}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

WORLD_MODEL_STATE_CKPT="${WORLD_MODEL_STATE_CKPT:-/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/rynn_backbone_dreamerv3_wm/rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed_goal_h5_epoch000_hidden_ddp_precomputed_bs96_nw2_gpu0123_viz_debug_20260509_091958/ckpt/latest.ckpt}"
RYNN_HIDDEN_DIR="${RYNN_HIDDEN_DIR:-/home/user01/liops/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export WORLD_MODEL_STATE_CKPT
export RYNN_HIDDEN_DIR

echo "[offline] GPUs=${CUDA_VISIBLE_DEVICES} batch_size_per_rank=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[offline] wm=${WORLD_MODEL_STATE_CKPT}"

exec bash scripts/train_dreamer_vla.sh \
  training.num_epochs="${NUM_EPOCHS:-100}" \
  training.max_train_steps="${MAX_TRAIN_STEPS:-100000}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  dataloader.persistent_workers=true \
  dataloader.prefetch_factor=1 \
  training.checkpoint_every=1
