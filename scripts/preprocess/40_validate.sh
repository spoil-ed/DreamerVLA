#!/usr/bin/env bash
# Validate one canonical OpenVLA input-token preprocessing artifact tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
if [[ "${LIBERO_SUITE}" == "${TASK}" ]]; then
  case "${TASK_NAME}" in
    openvla_onetraj_libero) LIBERO_SUITE="libero_goal" ;;
  esac
fi
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
if [[ "${ARTIFACT_NAME}" == "${TASK_NAME}" && "${TASK_NAME}" != "${LIBERO_SUITE}" ]]; then
  ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"
fi
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_remaining_reward"
OFT_HISTORY="${OFT_HISTORY:-1}"
INPUT_TOKEN_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_oft_input_token_embedding_vla_policy_h${OFT_HISTORY}"

python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${INPUT_TOKEN_DIR}" \
  reference_dir="${REWARD_DIR}" \
  match_reference_demos=true \
  match_reference_lengths=true \
  require_complete_attr=true \
  require_config=true
