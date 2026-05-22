#!/usr/bin/env bash
# Full data preparation pipeline:
# raw LIBERO HDF5 -> no-noop pixel HDF5 -> VLA/pretokenized data/configs.
#
# Optional:
#   PREPARE_LATENT_DATA=1 also generates the pi0 action-query hidden sidecar
#   used by the current action-hidden DreamerVLA route.
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Override defaults via env vars:
#   LIBERO_TASK_SUITE=libero_goal IMAGE_RESOLUTION=256 bash scripts/prepare_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_DIR="${SCRIPT_DIR}/preprocess"
source "${SCRIPT_DIR}/env_libero_goal.sh"

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export ACTION_HORIZON="${ACTION_HORIZON:-5}"
export TASK_NAME="${TASK_NAME:-goal}"
export LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-${TASK_NAME}}"
export PREPARE_LATENT_DATA="${PREPARE_LATENT_DATA:-0}"

TOTAL_STEPS=5
if [[ "${PREPARE_LATENT_DATA}" == "1" ]]; then
  TOTAL_STEPS=6
fi

echo "=== Step 1/${TOTAL_STEPS}: Filter no-ops from raw LIBERO HDF5 ==="
bash "${PREPROCESS_DIR}/processed_data_no_op.sh"

echo "=== Step 2/${TOTAL_STEPS}: Extract images / actions / states ==="
bash "${PREPROCESS_DIR}/processed_data_save_img_action_state_wrist.sh"

echo "=== Step 3/${TOTAL_STEPS}: Generate conversation JSONs ==="
bash "${PREPROCESS_DIR}/processed_data_generate_convs.sh"

echo "=== Step 4/${TOTAL_STEPS}: Pretokenize + build manifest ==="
bash "${PREPROCESS_DIR}/processed_data_pretokenize.sh"

echo "=== Step 5/${TOTAL_STEPS}: Write training YAML configs ==="
bash "${PREPROCESS_DIR}/prepare_train_configs.sh"

if [[ "${PREPARE_LATENT_DATA}" == "1" ]]; then
  echo "=== Step 6/${TOTAL_STEPS}: Precompute RynnVLA hidden/fullseq sidecar ==="
  bash "${PREPROCESS_DIR}/processed_data_rynn_hidden_fullseq.sh"
fi

echo "=== Data preparation complete ==="
