#!/usr/bin/env bash
# Full data preparation pipeline: raw LIBERO HDF5 → pretokenized WM training data.
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Override defaults via env vars:
#   LIBERO_TASK_SUITE=libero_goal IMAGE_RESOLUTION=256 bash scripts/prepare_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_DIR="${SCRIPT_DIR}/preprocess"

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export ACTION_HORIZON="${ACTION_HORIZON:-10}"
export TASK_NAME="${TASK_NAME:-goal}"

echo "=== Step 1/5: Filter no-ops from raw LIBERO HDF5 ==="
bash "${PREPROCESS_DIR}/processed_data_no_op.sh"

echo "=== Step 2/5: Extract images / actions / states ==="
bash "${PREPROCESS_DIR}/processed_data_save_img_action_state_wrist.sh"

echo "=== Step 3/5: Generate conversation JSONs ==="
bash "${PREPROCESS_DIR}/processed_data_generate_convs.sh"

echo "=== Step 4/5: Pretokenize + build manifest ==="
bash "${PREPROCESS_DIR}/processed_data_pretokenize.sh"

echo "=== Step 5/5: Write training YAML configs ==="
bash "${PREPROCESS_DIR}/prepare_train_configs.sh"

echo "=== Data preparation complete ==="
