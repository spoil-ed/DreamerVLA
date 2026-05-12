#!/usr/bin/env bash
# Canonical LIBERO-goal path registry for the pi0-style action-query VLA route.
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
export PI0_QUERY_VLA_STATE_CKPT="${PI0_QUERY_VLA_STATE_CKPT:-${PI0_QUERY_VLA_RUN_DIR}/ckpt/latest.ckpt}"

export VLA_STATE_CKPT="${PI0_QUERY_VLA_STATE_CKPT}"
export ENCODER_STATE_CKPT="${VLA_STATE_CKPT}"

export PI0_QUERY_HIDDEN_DIR="${PI0_QUERY_HIDDEN_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_pi0_query_h5_latest_fullseq}"
export RYNN_HIDDEN_DIR="${PI0_QUERY_HIDDEN_DIR}"
export RYNN_HIDDEN_FULLSEQ_DIR="${PI0_QUERY_HIDDEN_DIR}"
