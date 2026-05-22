#!/usr/bin/env bash
# SECONDARY pooled-hidden LIBERO-goal / RynnVLA path registry.
# Current mainline uses env_libero_goal_pi0_query.sh.
#
# Source this file from launch scripts when the experiment must keep the VLA
# base checkpoint, finetuned action head, hidden sidecar, and horizon aligned.
# Values are only set when the caller has not already provided an override.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  _DREAMERVLA_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export PROJECT_ROOT="$(cd "${_DREAMERVLA_ENV_DIR}/.." && pwd)"
  unset _DREAMERVLA_ENV_DIR
fi

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export ACTION_HORIZON="${ACTION_HORIZON:-5}"
export TIME_HORIZON="${TIME_HORIZON:-${ACTION_HORIZON}}"
export VLA_INIT_TAG="${VLA_INIT_TAG:-libero_goal}"

export VLA_INIT_CKPT="${VLA_INIT_CKPT:-${PROJECT_ROOT}/data/ckpts/VLA_model_256/libero_goal}"
export MODEL_PATH="${MODEL_PATH:-${VLA_INIT_CKPT}}"

export VLA_STATE_CKPT="${VLA_STATE_CKPT:-${PROJECT_ROOT}/data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_libero_goal_h5_20260508_060320/checkpoints/goal_h5_epoch000_train_vla_loss_1p323.ckpt}"
export ENCODER_STATE_CKPT="${ENCODER_STATE_CKPT:-${VLA_STATE_CKPT}}"

export HDF5_DIR="${HDF5_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256}"
export RYNN_HIDDEN_DIR="${RYNN_HIDDEN_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000}"
export RYNN_HIDDEN_FULLSEQ_DIR="${RYNN_HIDDEN_FULLSEQ_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000_fullseq}"

export DREAMERVLA_UNIFIED_VLA_TAG="${DREAMERVLA_UNIFIED_VLA_TAG:-libero_goal_epoch000_h5_horizon5}"
