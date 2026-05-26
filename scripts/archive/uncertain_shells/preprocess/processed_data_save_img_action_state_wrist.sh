#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$ROOT_DIR/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/data/processed_data/${LIBERO_TASK_SUITE}_image_state_action_t_${IMAGE_RESOLUTION}}"

mkdir -p "$(dirname "$SAVE_DIR")"

python "$ROOT_DIR/src/utils/libero_utils/regenerate_libero_dataset_save_img_action_state_wrist.py" \
    --libero_task_suite "$LIBERO_TASK_SUITE" \
    --image_resolution "$IMAGE_RESOLUTION" \
    --raw_data_dir "$RAW_DATA_DIR" \
    --save_dir "$SAVE_DIR"
