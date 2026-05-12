#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="/home/user01/miniconda3/envs/dreamervla/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_TAG="${CUDA_VISIBLE_DEVICES//,/}"
TASK_IDS="${TASK_IDS:-0}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-8}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-32}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-200000}"
MIN_REPLAY="${MIN_REPLAY:-64}"
TRAIN_EVERY="${TRAIN_EVERY:-4}"
UPDATES_PER_TRAIN="${UPDATES_PER_TRAIN:-1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
SEED="${SEED:-7}"
RUN_TAG="${RUN_TAG:-online_rynn_meanpool_dreamer_actor_task${TASK_IDS//,/}_bs${ONLINE_BATCH_SIZE}_gpu${GPU_TAG}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/outputs/dreamervla_online/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

WORLD_MODEL_STATE_CKPT="${WORLD_MODEL_STATE_CKPT:-/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/rynn_backbone_dreamerv3_wm/rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed_goal_h5_epoch000_hidden_ddp_precomputed_bs96_nw2_gpu0123_viz_debug_20260509_091958/ckpt/latest.ckpt}"
VLA_CKPT_PATH="${VLA_CKPT_PATH:-/home/user01/liops/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_goal}"
ENCODER_STATE_CKPT="${ENCODER_STATE_CKPT:-/home/user01/liops/workspace/DreamerVLA/data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_libero_goal_h5_20260508_060320/checkpoints/goal_h5_epoch000_train_vla_loss_1p323.ckpt}"

echo "[online] GPU=${CUDA_VISIBLE_DEVICES} task_ids=${TASK_IDS} batch=${ONLINE_BATCH_SIZE} seq=${SEQUENCE_LENGTH}"
echo "[online] out=${OUT_DIR}"
echo "[online] wm=${WORLD_MODEL_STATE_CKPT}"

exec python scripts/train_online_rynn_meanpool_dreamer_actor.py \
  --out-dir "${OUT_DIR}" \
  --world-model-ckpt "${WORLD_MODEL_STATE_CKPT}" \
  --vla-ckpt-path "${VLA_CKPT_PATH}" \
  --encoder-state-ckpt "${ENCODER_STATE_CKPT}" \
  --task-suite "${TASK_SUITE:-libero_goal}" \
  --task-ids "${TASK_IDS}" \
  --seed "${SEED}" \
  --max-env-steps "${MAX_ENV_STEPS}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --batch-size "${ONLINE_BATCH_SIZE}" \
  --replay-size "${REPLAY_SIZE:-20000}" \
  --min-replay "${MIN_REPLAY}" \
  --train-every "${TRAIN_EVERY}" \
  --updates-per-train "${UPDATES_PER_TRAIN}" \
  --save-every "${SAVE_EVERY}" \
  --log-every "${LOG_EVERY}"
