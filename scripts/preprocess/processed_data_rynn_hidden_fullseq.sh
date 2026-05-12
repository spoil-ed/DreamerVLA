#!/usr/bin/env bash
# Step 6 for DreamerVLA data preparation:
# precomputed RynnVLA pooled hidden + full token hidden sequence sidecar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${ROOT_DIR}/scripts/env_libero_goal.sh"

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export HDF5_DIR="${HDF5_DIR:-${ROOT_DIR}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
export OUT_DIR="${OUT_DIR:-${RYNN_HIDDEN_FULLSEQ_DIR}}"
export SAVE_ACTOR_SEQUENCE="${SAVE_ACTOR_SEQUENCE:-1}"

bash "${ROOT_DIR}/scripts/preprocess_rynn_pixel_hidden.sh"
