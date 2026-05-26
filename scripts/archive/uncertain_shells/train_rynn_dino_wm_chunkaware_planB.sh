#!/usr/bin/env bash
# Pretrain the chunk-aware Rynn-DINO WM with the true Plan B objective:
# chunk-as-one-input dynamics (K-step prediction in a SINGLE transformer pass,
# future obs slots filled by the learned `mask_obs_token`).
#
# Until this WM is (re)trained with chunk_loss, the mask_obs_token is randomly
# initialized and any imagined rollout via predict_next_chunk is essentially
# noise — so this script MUST be run before any chunk-WM-dependent downstream
# training (online WMPO outcome PPO etc.).
#
# Usage:
#   # Warm-start from the most recent step-level chunkaware_pinned ckpt (default)
#   WARM_START=1 CUDA_VISIBLE_DEVICES=4 \
#     bash scripts/train_rynn_dino_wm_chunkaware_planB.sh
#
#   # Fresh train (no warm-start) on GPU 4
#   CUDA_VISIBLE_DEVICES=4 bash scripts/train_rynn_dino_wm_chunkaware_planB.sh
#
#   # Multi-GPU DDP
#   DDP=1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     bash scripts/train_rynn_dino_wm_chunkaware_planB.sh
#
#   # Smoke (1 step) for sanity
#   WM_SMOKE=1 CUDA_VISIBLE_DEVICES=4 \
#     bash scripts/train_rynn_dino_wm_chunkaware_planB.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WARM_START_CKPT_DEFAULT="/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_pinned/step_00017000.ckpt"

EXTRA_OVERRIDES=()
if [[ "${WARM_START:-0}" == "1" ]]; then
  WARM_START_CKPT="${WARM_START_CKPT:-${WARM_START_CKPT_DEFAULT}}"
  if [[ ! -f "${WARM_START_CKPT}" ]]; then
    echo "ERROR: WARM_START=1 but ckpt not found: ${WARM_START_CKPT}" >&2
    exit 2
  fi
  EXTRA_OVERRIDES+=(
    "training.resume=true"
    "training.resume_path=${WARM_START_CKPT}"
    "training.resume_strict=false"
    "training.resume_skip_optimizer=true"
  )
  echo "[chunkaware-planB] warm-start from: ${WARM_START_CKPT}"
fi

CONFIG_NAME="${CONFIG_NAME:-rynn_dino_wm_chunkaware_libero_goal_planB}" \
  bash "${SCRIPT_DIR}/train_wm.sh" "${EXTRA_OVERRIDES[@]}" "$@"
