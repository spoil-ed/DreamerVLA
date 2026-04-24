#!/usr/bin/env bash
# LIBERO rollout evaluation for a saved VLA checkpoint (no training).
#
# Usage:
#   conda activate wmpo
#   bash scripts/eval_libero_vla.sh \
#     eval.ckpt_path=data/outputs/pretokenize_vla/checkpoints/epoch=013-train_vla_loss=1.984.ckpt \
#     eval.task_suite_name=libero_goal \
#     eval.num_episodes_per_task=10
#
# Rollout must run on a single process (single GPU); the underlying LIBERO
# benchmark does not support sharded inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_NAME="${CONFIG_NAME:-eval_libero_vla}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/eval_libero_vla}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"

echo "Run output dir: ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@"
