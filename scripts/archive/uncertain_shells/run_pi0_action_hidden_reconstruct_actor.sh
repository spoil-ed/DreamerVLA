#!/usr/bin/env bash
# Joint reconstruction + actor training for the current DreamerVLA route:
#
#   RSSM(h,z) -> hidden_decoder reconstructs pi0 action hidden
#             -> Pi0ActionHiddenActor -> action
#
# Unlike the earlier actor-only sweep, this keeps training.run_wm_phase=true so
# every batch continues to optimize action-hidden reconstruction loss.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/lib/output_layout.sh"

CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_pi0_action_hidden_head_actor}"
PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pi0_action_hidden_reconstruct_actor_${TIMESTAMP}_gpu${CUDA_VISIBLE_DEVICES//,/}}"
OUT_DIR="${OUT_DIR:-$(output_layout_path "${PROJECT_ROOT}" dreamervla pi0_action_hidden_actor reconstruct_actor "${RUN_NAME}")}"
OLD_DREAMERVLA_CKPT="${OLD_DREAMERVLA_CKPT:-${PROJECT_ROOT}/data/outputs/dreamervla/dreamervla_action_hidden_actor_20260512_gpu4567_wm10000/ckpt/latest.ckpt}"

mkdir -p "${OUT_DIR}"

echo "=== DreamerVLA action-hidden reconstruction actor training ==="
echo "old_dreamervla_ckpt=${OLD_DREAMERVLA_CKPT}"
echo "out_dir=${OUT_DIR}"
echo "num_epochs=${NUM_EPOCHS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo

CONFIG_NAME="${CONFIG_NAME}" \
DREAMERVLA_STATE_CKPT="${OLD_DREAMERVLA_CKPT}" \
PYTHON="${PYTHON}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NUM_GPUS="${NUM_GPUS}" \
OUT_DIR="${OUT_DIR}" \
bash scripts/train_dreamer_vla.sh \
  training.run_wm_phase=true \
  training.run_actor_critic_phase=true \
  training.num_epochs="${NUM_EPOCHS}" \
  training.checkpoint_every=1 \
  policy.adapter_type="${POLICY_ADAPTER_TYPE:-residual_mlp}" \
  policy.freeze_output_projection="${FREEZE_OUTPUT_PROJECTION:-false}" \
  "$@" 2>&1 | tee "${OUT_DIR}/train.log"
