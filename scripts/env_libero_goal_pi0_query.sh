#!/usr/bin/env bash
# CURRENT MAINLINE LIBERO-goal path registry for the pi0-style action-query VLA route.
#
# Source this after env_libero_goal.sh, or source it directly.  It keeps the
# finetuned pi0-query VLA ckpt, hidden sidecar, WM, and DreamerVLA actor type
# aligned so this route cannot silently fall back to the legacy action head.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  _DREAMERVLA_PI0_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export PROJECT_ROOT="$(cd "${_DREAMERVLA_PI0_ENV_DIR}/.." && pwd)"
  unset _DREAMERVLA_PI0_ENV_DIR
fi

source "${PROJECT_ROOT}/scripts/env_libero_goal.sh"

export ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-pi0_query}"
export DREAMERVLA_UNIFIED_VLA_TAG="libero_goal_pi0_query_h5_horizon5"

export PI0_QUERY_VLA_RUN_DIR="${PI0_QUERY_VLA_RUN_DIR:-${PROJECT_ROOT}/data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_pi0_query_libero_goal_pi0_query_h5_20260510_205352}"
export PI0_QUERY_VLA_STATE_CKPT="${PI0_QUERY_VLA_STATE_CKPT:-${PROJECT_ROOT}/data/ckpts/pi0_query_vla_libero_goal/epoch003_train_vla_loss1.255_success8of10.ckpt}"

export VLA_STATE_CKPT="${PI0_QUERY_VLA_STATE_CKPT}"
export ENCODER_STATE_CKPT="${VLA_STATE_CKPT}"

# Canonical pi0 action-hidden observation input.  Keep this aligned with the
# existing preprocessed sidecar:
#   Finish the task: {prompt}. + <|state|> + his_2 two-view images + rot180.
export PI0_QUERY_PROMPT_STYLE="${PI0_QUERY_PROMPT_STYLE:-vla_policy}"
export PI0_QUERY_HISTORY="${PI0_QUERY_HISTORY:-2}"
export PI0_QUERY_INCLUDE_STATE="${PI0_QUERY_INCLUDE_STATE:-1}"
export PI0_QUERY_ROTATE_IMAGES_180="${PI0_QUERY_ROTATE_IMAGES_180:-1}"
export PI0_QUERY_OBS_HIDDEN_SOURCE="${PI0_QUERY_OBS_HIDDEN_SOURCE:-action_query}"

export PI0_QUERY_HIDDEN_DIR="${PI0_QUERY_HIDDEN_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_pi0_action_hidden_vla_policy_h2}"
export RYNN_HIDDEN_DIR="${PI0_QUERY_HIDDEN_DIR}"
export RYNN_HIDDEN_FULLSEQ_DIR="${PI0_QUERY_HIDDEN_DIR}"

export PI06_REMAINING_REWARD_HDF5_DIR="${PI06_REMAINING_REWARD_HDF5_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward}"
if [[ "${USE_PI06_REMAINING_REWARD:-1}" == "1" ]]; then
  export HDF5_DIR="${PI06_REMAINING_REWARD_HDF5_DIR}"
fi

export PI0_ACTION_HIDDEN_DIR="${PI0_ACTION_HIDDEN_DIR:-${PI0_QUERY_HIDDEN_DIR}}"
