#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
BASE_DIR="${BASE_DIR:-$ROOT_DIR/data/processed_data/libero_goal_image_state_action_t_${IMAGE_RESOLUTION}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/processed_data/convs}"

cd "$ROOT_DIR/src/preprocess"

python action_state_model_conv_generation.py \
    --base_dir "$BASE_DIR" \
    --his 2 \
    --len_action "$ACTION_HORIZON" \
    --task_name "$LIBERO_TASK_NAME" \
    --resolution "$IMAGE_RESOLUTION" \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --output_dir "$OUTPUT_DIR"
