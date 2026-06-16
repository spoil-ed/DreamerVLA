#!/usr/bin/env bash
# Online cotrain — Scheme 1 (backbone-latent WM). NOTE: online env rollout for
# backbone_latent is NOT wired (DreamerVLAOnlineTrainEnv emits action_query only);
# this entry raises a clear error and points to the runnable OFFLINE input-token
# path below. See docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_backbone_latent_world_model.md
#
# Runnable offline backbone-latent path:
#   bash scripts/train_dreamervla.sh \
#     experiment=dreamervla_oft_dino_wm_wmpo_outcome_input_tokens task=OpenVLA_Onetraj_LIBERO \
#     gpus=0 ngpu=1 -- init.world_model_state_ckpt=/abs/wm.ckpt init.classifier_state_ckpt=/abs/cls.ckpt
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"  # use cwd LIBERO paths, not stale ~/.libero
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${DVLA_ROOT}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
torchrun --standalone --nproc_per_node="${NUM_GPUS:-1}" \
  -m dreamervla.train experiment=online_cotrain_oft_backbone_latent "$@"
