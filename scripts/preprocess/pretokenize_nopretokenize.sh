#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK_NAME="${TASK_NAME:-goal}"
export LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-${TASK_NAME}}"

bash "$SCRIPT_DIR/processed_data_no_op.sh"
bash "$SCRIPT_DIR/processed_data_save_img_action_state_wrist.sh"
bash "$SCRIPT_DIR/processed_data_generate_convs.sh"
bash "$SCRIPT_DIR/processed_data_pretokenize.sh"
bash "$SCRIPT_DIR/prepare_train_configs.sh"
