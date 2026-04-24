#!/usr/bin/env bash
# VLA pretokenize SFT training with hyperparameters aligned to RynnVLA-002 official recipe.
#
# Aligned overrides vs default config (see docs for rationale):
#   - loss:   action coef = 10, token coef = 1
#   - optim:  AdamW, lr=5e-6, betas=(0.9,0.999), wd=0.1
#   - warmup: 50 steps (~1% of epoch, matching rynn)
# Also relies on source-code edits:
#   - src/utils/optim.py: AdamW support
#   - src/models/encoder/rynnvla_encoder.py: att_mask=True (custom block mask)
#
# Usage:
#   conda activate wmpo
#   bash scripts/pretokenize_train_vla.sh
#
# Override via env vars:
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/pretokenize_train_vla.sh
#   CONFIG_NAME=pretokenize_vla_libero_10 bash scripts/pretokenize_train_vla.sh
# Override/add any hydra arg as trailing args:
#   bash scripts/pretokenize_train_vla.sh training.num_epochs=5 optim.vla.lr=1e-6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_vla_libero_10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/pretokenize_vla}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Run output dir: ${OUT_DIR}"
echo "GPUs:           ${CUDA_VISIBLE_DEVICES}  (nproc_per_node=${NUM_GPUS})"
echo "Config:         ${CONFIG_NAME}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  +training.vla_action_loss_coef=10 \
  +training.vla_token_loss_coef=1 \
  training.lr_warmup_steps=50 \
  optim.vla.name=adamw \
  optim.vla.lr=5.0e-6 \
  optim.vla.betas=[0.9,0.999] \
  optim.vla.weight_decay=0.1 \
  training.out_dir="${OUT_DIR}" \
  training.resume=false \
  "$@"
