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
#   CONFIG_NAME=pretokenize_vla_libero_goal bash scripts/pretokenize_train_vla.sh
#   VLA_INIT_TAG=libero_goal bash scripts/pretokenize_train_vla.sh
#   VLA_INIT_CKPT=/path/to/VLA_model_256/libero_goal bash scripts/pretokenize_train_vla.sh
#   ACTION_HORIZON=5 bash scripts/pretokenize_train_vla.sh
#   ACTION_HEAD_TYPE=pi0_query CONFIG_NAME=pretokenize_vla_libero_goal_pi0_query bash scripts/pretokenize_train_vla.sh
#   VLA_TOKEN_LOSS_COEF=0 VLA_ACTION_LOSS_COEF=10 bash scripts/pretokenize_train_vla.sh
# Override/add any hydra arg as trailing args:
#   bash scripts/pretokenize_train_vla.sh training.num_epochs=5 optim.vla.lr=1e-6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/env_libero_goal.sh"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_vla_libero_goal}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
VLA_INIT_TAG="${VLA_INIT_TAG:-libero_goal}"
VLA_INIT_CKPT="${VLA_INIT_CKPT:-}"
if [[ -n "${VLA_INIT_TAG}" && -z "${VLA_INIT_CKPT}" ]]; then
  VLA_INIT_CKPT="${PROJECT_ROOT}/data/ckpts/VLA_model_256/${VLA_INIT_TAG}"
fi
ACTION_HORIZON="${ACTION_HORIZON:-}"
if [[ -z "${ACTION_HORIZON}" ]]; then
  case "${VLA_INIT_TAG}" in
    libero_goal|libero_object)
      ACTION_HORIZON=5
      ;;
    libero_10|libero_spatial)
      ACTION_HORIZON=10
      ;;
  esac
fi
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/vla/pretokenize_vla}"
RUN_NAME="${RUN_NAME:-${CONFIG_NAME}${VLA_INIT_TAG:+_${VLA_INIT_TAG}}${ACTION_HEAD_TYPE:+_${ACTION_HEAD_TYPE}}${ACTION_HORIZON:+_h${ACTION_HORIZON}}_${TIMESTAMP}}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
VLA_ACTION_LOSS_COEF="${VLA_ACTION_LOSS_COEF:-10}"
VLA_TOKEN_LOSS_COEF="${VLA_TOKEN_LOSS_COEF:-1}"

INIT_OVERRIDES=()
if [[ -n "${VLA_INIT_CKPT}" ]]; then
  INIT_OVERRIDES+=("init.vla_ckpt_path=${VLA_INIT_CKPT}")
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  INIT_OVERRIDES+=("encoder.time_horizon=${ACTION_HORIZON}")
  INIT_OVERRIDES+=("dataset.action_horizon=${ACTION_HORIZON}")
  INIT_OVERRIDES+=("dataset_val_ind.action_horizon=${ACTION_HORIZON}")
  INIT_OVERRIDES+=("dataset_val_ood.action_horizon=${ACTION_HORIZON}")
fi
if [[ -n "${ACTION_HEAD_TYPE}" ]]; then
  INIT_OVERRIDES+=("encoder.action_head_type=${ACTION_HEAD_TYPE}")
fi

echo "Run output dir: ${OUT_DIR}"
echo "GPUs:           ${CUDA_VISIBLE_DEVICES}  (nproc_per_node=${NUM_GPUS})"
echo "Config:         ${CONFIG_NAME}"
if [[ -n "${VLA_INIT_TAG}" ]]; then
  echo "VLA init tag:   ${VLA_INIT_TAG}"
fi
if [[ -n "${VLA_INIT_CKPT}" ]]; then
  echo "VLA init ckpt:  ${VLA_INIT_CKPT}"
fi
if [[ -n "${ACTION_HORIZON}" ]]; then
  echo "Action horizon: ${ACTION_HORIZON}"
fi
if [[ -n "${ACTION_HEAD_TYPE}" ]]; then
  echo "Action head:    ${ACTION_HEAD_TYPE}"
fi
echo "Action coef:    ${VLA_ACTION_LOSS_COEF}"
echo "Token coef:     ${VLA_TOKEN_LOSS_COEF}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  training.vla_action_loss_coef="${VLA_ACTION_LOSS_COEF}" \
  training.vla_token_loss_coef="${VLA_TOKEN_LOSS_COEF}" \
  training.lr_warmup_steps=50 \
  optim.vla.name=adamw \
  optim.vla.lr=5.0e-6 \
  optim.vla.betas=[0.9,0.999] \
  optim.vla.weight_decay=0.1 \
  training.out_dir="${OUT_DIR}" \
  training.resume=false \
  "${INIT_OVERRIDES[@]}" \
  "$@"
