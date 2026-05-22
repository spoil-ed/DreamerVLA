#!/usr/bin/env bash
# End-to-end pi0-query hidden pipeline for LIBERO-goal:
#   1. use the pi0-query VLA action head ckpt
#   2. precompute pi0 action-query hidden sidecar
#   3. train the action-hidden DreamerV3 world model
#   4. train the pi0 action-hidden DreamerVLA actor.
#
# Default mode prints the exact commands without launching long jobs:
#   bash scripts/run_pi0_query_hidden_pipeline.sh
#
# Execute one stage:
#   PIPELINE_STAGE=preprocess bash scripts/run_pi0_query_hidden_pipeline.sh
#   PIPELINE_STAGE=wm WORLD_MODEL_STATE_CKPT=... bash scripts/run_pi0_query_hidden_pipeline.sh
#   PIPELINE_STAGE=dreamervla WORLD_MODEL_STATE_CKPT=... bash scripts/run_pi0_query_hidden_pipeline.sh
#
# Execute sequentially:
#   PIPELINE_STAGE=all bash scripts/run_pi0_query_hidden_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/env_libero_goal_pi0_query.sh"

PIPELINE_STAGE="${PIPELINE_STAGE:-commands}" # commands | preprocess | wm | dreamervla | all
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"

PREPROCESS_GPUS="${PREPROCESS_GPUS:-6,7}"
PREPROCESS_NUM_GPUS="${PREPROCESS_NUM_GPUS:-2}"
WM_GPUS="${WM_GPUS:-6,7}"
WM_NUM_GPUS="${WM_NUM_GPUS:-2}"
DREAMERVLA_GPUS="${DREAMERVLA_GPUS:-6,7}"
DREAMERVLA_NUM_GPUS="${DREAMERVLA_NUM_GPUS:-2}"

PREPROCESS_MASTER_PORT="${PREPROCESS_MASTER_PORT:-29551}"
WM_MASTER_PORT="${WM_MASTER_PORT:-29552}"
DREAMERVLA_MASTER_PORT="${DREAMERVLA_MASTER_PORT:-29553}"

PI0_QUERY_VLA_STATE_CKPT="${PI0_QUERY_VLA_STATE_CKPT:-${VLA_STATE_CKPT}}"
PI0_QUERY_HIDDEN_DIR="${PI0_QUERY_ACTION_HIDDEN_DIR:-${PI0_ACTION_HIDDEN_DIR}}"
PI0_QUERY_WM_OUT_DIR="${PI0_QUERY_WM_OUT_DIR:-${PROJECT_ROOT}/data/outputs/worldmodel/action_hidden_dreamerv3_wm/pi0_action_hidden_wm_${PIPELINE_ID}}"
PI0_QUERY_DREAMERVLA_OUT_DIR="${PI0_QUERY_DREAMERVLA_OUT_DIR:-${PROJECT_ROOT}/data/outputs/dreamervla/pi0_action_hidden_dreamervla_${PIPELINE_ID}}"
WORLD_MODEL_STATE_CKPT="${WORLD_MODEL_STATE_CKPT:-${PI0_QUERY_WM_OUT_DIR}/ckpt/latest.ckpt}"

ACTION_HORIZON="${ACTION_HORIZON:-5}"
TIME_HORIZON="${TIME_HORIZON:-${ACTION_HORIZON}}"
ACTOR_SEQUENCE_LENGTH="${ACTOR_SEQUENCE_LENGTH:-640}"
OBS_HIDDEN_SOURCE="${OBS_HIDDEN_SOURCE:-action_query}"
RYNN_WM_OBS_DIM="${RYNN_WM_OBS_DIM:-5120}"

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  "$@"
}

preprocess_cmd=(
  env
  "CUDA_VISIBLE_DEVICES=${PREPROCESS_GPUS}"
  "NUM_GPUS=${PREPROCESS_NUM_GPUS}"
  "MASTER_PORT=${PREPROCESS_MASTER_PORT}"
  "ACTION_HEAD_TYPE=pi0_query"
  "VLA_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "ENCODER_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "SAVE_ACTOR_SEQUENCE=${SAVE_ACTOR_SEQUENCE:-0}"
  "OBS_HIDDEN_SOURCE=${OBS_HIDDEN_SOURCE}"
  "PROMPT_STYLE=${PI0_QUERY_PROMPT_STYLE}"
  "HISTORY=${PI0_QUERY_HISTORY}"
  "INCLUDE_STATE=${PI0_QUERY_INCLUDE_STATE}"
  "ROTATE_IMAGES_180=${PI0_QUERY_ROTATE_IMAGES_180}"
  "SAVE_ACTION_HIDDEN=${SAVE_ACTION_HIDDEN:-1}"
  "OUT_DIR=${PI0_QUERY_HIDDEN_DIR}"
  "ACTION_HORIZON=${ACTION_HORIZON}"
  "TIME_HORIZON=${TIME_HORIZON}"
  "CHUNK_SIZE=${CHUNK_SIZE:-16}"
  "OUTPUT_DTYPE=${OUTPUT_DTYPE:-float16}"
  "OVERWRITE=${OVERWRITE_HIDDEN:-0}"
  bash scripts/preprocess_rynn_pixel_hidden.sh
)

wm_cmd=(
  env
  "CUDA_VISIBLE_DEVICES=${WM_GPUS}"
  "NUM_GPUS=${WM_NUM_GPUS}"
  "MASTER_PORT=${WM_MASTER_PORT}"
  "OUT_DIR=${PI0_QUERY_WM_OUT_DIR}"
  "CONFIG_NAME=rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed"
  "ACTION_HIDDEN_DIR=${PI0_QUERY_HIDDEN_DIR}"
  "VLA_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "ENCODER_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "ACTION_HORIZON=${ACTION_HORIZON}"
  "TIME_HORIZON=${TIME_HORIZON}"
  "OBS_HIDDEN_SOURCE=${OBS_HIDDEN_SOURCE}"
  "ACTION_HIDDEN_EXPECTED_OBS_HIDDEN_SOURCE=action_query"
  "ACTION_HIDDEN_EXPECTED_PROMPT_STYLE=vla_policy"
  "ACTION_HIDDEN_EXPECTED_HISTORY=2"
  "ACTION_HIDDEN_EXPECTED_INCLUDE_STATE=true"
  "ACTION_HIDDEN_EXPECTED_ROTATE_IMAGES_180=true"
  "ACTION_HIDDEN_WM_OBS_DIM=${RYNN_WM_OBS_DIM}"
  "LOAD_ACTOR_SEQUENCE=false"
  "ACTOR_SEQUENCE_LENGTH=0"
  "FULL_HIDDEN_REC_SCALE=0.0"
  "BATCH_SIZE=${WM_BATCH_SIZE:-96}"
  "NUM_WORKERS=${WM_NUM_WORKERS:-2}"
  bash scripts/train_pi0_action_hidden_dreamerv3_wm.sh
  "dataset.hidden_dir=${PI0_QUERY_HIDDEN_DIR}"
  "dataset.expected_encoder_state_ckpt=${PI0_QUERY_VLA_STATE_CKPT}"
  "dataset.expected_action_head_type=pi0_query"
  "dataset.expected_obs_hidden_source=action_query"
  "dataset.expected_prompt_style=vla_policy"
  "dataset.expected_history=2"
  "dataset.expected_include_state=true"
  "dataset.expected_rotate_images_180=true"
)

dreamervla_cmd=(
  env
  "CUDA_VISIBLE_DEVICES=${DREAMERVLA_GPUS}"
  "NUM_GPUS=${DREAMERVLA_NUM_GPUS}"
  "MASTER_PORT=${DREAMERVLA_MASTER_PORT}"
  "OUT_DIR=${PI0_QUERY_DREAMERVLA_OUT_DIR}"
  "CONFIG_NAME=dreamer_vla_libero_goal_pi0_action_hidden_head_actor"
  "ACTION_HIDDEN_DIR=${PI0_QUERY_HIDDEN_DIR}"
  "RYNN_HIDDEN_DIR=${PI0_QUERY_HIDDEN_DIR}"
  "VLA_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "ENCODER_STATE_CKPT=${PI0_QUERY_VLA_STATE_CKPT}"
  "WORLD_MODEL_STATE_CKPT=${WORLD_MODEL_STATE_CKPT}"
  "ACTION_HORIZON=${ACTION_HORIZON}"
  "TIME_HORIZON=${TIME_HORIZON}"
  "BATCH_SIZE=${DREAMERVLA_BATCH_SIZE:-10}"
  "NUM_WORKERS=${DREAMERVLA_NUM_WORKERS:-2}"
  bash scripts/train_dreamer_vla.sh
  "dataset.hidden_dir=${PI0_QUERY_HIDDEN_DIR}"
  "dataset.expected_encoder_state_ckpt=${PI0_QUERY_VLA_STATE_CKPT}"
  "+dataset.expected_action_head_type=pi0_query"
  "+dataset.expected_obs_hidden_source=action_query"
  "+dataset.expected_prompt_style=vla_policy"
  "+dataset.expected_history=2"
  "+dataset.expected_include_state=true"
  "+dataset.expected_rotate_images_180=true"
  "+policy.action_head_type=pi0_query"
  "policy.init_action_head_ckpt=${PI0_QUERY_VLA_STATE_CKPT}"
)

echo "=== pi0-query hidden pipeline ==="
echo "stage:              ${PIPELINE_STAGE}"
echo "pipeline_id:        ${PIPELINE_ID}"
echo "vla_state_ckpt:     ${PI0_QUERY_VLA_STATE_CKPT}"
echo "hidden_dir:         ${PI0_QUERY_HIDDEN_DIR}"
echo "wm_out_dir:         ${PI0_QUERY_WM_OUT_DIR}"
echo "wm_state_ckpt:      ${WORLD_MODEL_STATE_CKPT}"
echo "dreamervla_out_dir: ${PI0_QUERY_DREAMERVLA_OUT_DIR} (follow-up; not launched by action-hidden all)"
echo

case "${PIPELINE_STAGE}" in
  commands)
    echo "[1/3] Precompute pi0-query hidden sidecar:"
    print_cmd "${preprocess_cmd[@]}"
    echo
    echo "[2/3] Train pi0-query hidden world model:"
    print_cmd "${wm_cmd[@]}"
    echo
    echo "[3/3] DreamerVLA actor stage:"
    print_cmd "${dreamervla_cmd[@]}"
    ;;
  preprocess)
    run_cmd "${preprocess_cmd[@]}"
    ;;
  wm)
    run_cmd "${wm_cmd[@]}"
    ;;
  dreamervla)
    run_cmd "${dreamervla_cmd[@]}"
    ;;
  all)
    run_cmd "${preprocess_cmd[@]}"
    run_cmd "${wm_cmd[@]}"
    run_cmd "${dreamervla_cmd[@]}"
    ;;
  *)
    echo "Unknown PIPELINE_STAGE='${PIPELINE_STAGE}'. Use commands, preprocess, wm, dreamervla, or all." >&2
    exit 2
    ;;
esac
