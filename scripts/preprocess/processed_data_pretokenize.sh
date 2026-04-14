#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ROOT_DIR/data/processed_data}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-32}"
TASK_NAME="${TASK_NAME:-goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
IMG_NAMES=(${IMG_NAMES:-imgs_third_view imgs_wrist})

cd "$ROOT_DIR/src/preprocess"

python pretoken_state_action_model.py \
    --task "$TASK_NAME" \
    --resolution "$IMAGE_RESOLUTION" \
    --with_state \
    --img_names "${IMG_NAMES[@]}" \
    --his 2 \
    --len_action "$ACTION_HORIZON" \
    --num_procs "$PRETOKENIZE_PROCS" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --in_filename_dir "$PROCESSED_DATA_ROOT/convs" \
    --out_root "$PROCESSED_DATA_ROOT/tokens"

bash concate_record_libero.sh "$PROCESSED_DATA_ROOT/tokens"

mkdir -p "$PROCESSED_DATA_ROOT/concate_tokens"

python concate_action_world_model_data_libero.py \
    --source_dir_patterns libero_${TASK_NAME}_his_2_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION} \
    --all_patterns libero_${TASK_NAME}_his_2_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION} \
    --processed_data_root "$PROCESSED_DATA_ROOT"
