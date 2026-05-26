#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$ROOT_DIR/data/libero/datasets/${LIBERO_TASK_SUITE}}"
TARGET_DIR="${TARGET_DIR:-$ROOT_DIR/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"

mkdir -p "$(dirname "$TARGET_DIR")"

python "$ROOT_DIR/src/utils/libero_utils/regenerate_libero_dataset_filter_no_op.py" \
    --libero_task_suite "$LIBERO_TASK_SUITE" \
    --libero_raw_data_dir "$RAW_DATA_DIR" \
    --libero_target_dir "$TARGET_DIR" \
    --image_resolution "$IMAGE_RESOLUTION"
