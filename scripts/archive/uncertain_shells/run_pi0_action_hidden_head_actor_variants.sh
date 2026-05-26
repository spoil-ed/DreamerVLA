#!/usr/bin/env bash
# Sequential sweep for the newer DreamerVLA route:
#   old action-hidden DreamerVLA ckpt -> reconstructed action hidden -> pi0 action head actor.
#
# The old checkpoint provides world_model / critic / target_critic / return_tracker.
# The policy is intentionally re-created because the actor class changed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_pi0_action_hidden_head_actor}"
PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-${PROJECT_ROOT}/data/outputs/dreamervla/pi0_action_hidden_head_actor_variants_${TIMESTAMP}}"
OLD_DREAMERVLA_CKPT="${OLD_DREAMERVLA_CKPT:-${PROJECT_ROOT}/data/outputs/dreamervla/dreamervla_action_hidden_actor_20260512_gpu4567_wm10000/ckpt/latest.ckpt}"

run_variant() {
  local name="$1"
  shift
  local out_dir="${BASE_OUT}/${name}"
  mkdir -p "${out_dir}"
  echo "=== [$(date '+%F %T')] ${name} ==="
  echo "old_ckpt=${OLD_DREAMERVLA_CKPT}"
  echo "out_dir=${out_dir}"
  CONFIG_NAME="${CONFIG_NAME}" \
  DREAMERVLA_STATE_CKPT="${OLD_DREAMERVLA_CKPT}" \
  PYTHON="${PYTHON}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_GPUS="${NUM_GPUS}" \
  OUT_DIR="${out_dir}" \
  bash scripts/train_dreamer_vla.sh \
    training.run_wm_phase=false \
    training.run_actor_critic_phase=true \
    training.num_epochs="${NUM_EPOCHS}" \
    training.checkpoint_every=1 \
    "$@" 2>&1 | tee "${out_dir}/train.log"
}

run_variant residual_frozen_head \
  policy.adapter_type=residual_mlp \
  policy.freeze_output_projection=true

run_variant residual_unfrozen_head \
  policy.adapter_type=residual_mlp \
  policy.freeze_output_projection=false

run_variant identity_unfrozen_head \
  policy.adapter_type=identity \
  policy.freeze_output_projection=false

echo "=== [$(date '+%F %T')] all variants finished: ${BASE_OUT} ==="
