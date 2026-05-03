#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
TASK_NAME="${TASK_NAME:-goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/data/configs/${LIBERO_TASK_SUITE}}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ROOT_DIR/data/processed_data}"

mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/concate_tokens/libero_${TASK_NAME}_his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize_val_ind.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/tokens/libero_${TASK_NAME}_his_1_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize_val_ood.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/tokens/libero_${TASK_NAME}_his_1_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_nopretokenize.yaml" <<EOF
META:
  split: all
  libero_task_suite: $LIBERO_TASK_SUITE
  raw_data_dir: '$PROCESSED_DATA_ROOT/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}'
action_model:
  len_action: $ACTION_HORIZON
  his: 1
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize_seq.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/concate_tokens/libero_${TASK_NAME}_his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize_seq_val_ind.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/tokens/libero_${TASK_NAME}_his_1_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "$CONFIG_DIR/his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize_seq_val_ood.yaml" <<EOF
META:
  - path: '$PROCESSED_DATA_ROOT/tokens/libero_${TASK_NAME}_his_1_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

echo "Wrote configs to $CONFIG_DIR"
