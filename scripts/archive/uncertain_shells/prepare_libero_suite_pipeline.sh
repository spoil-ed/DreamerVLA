#!/usr/bin/env bash
# Parameterized data pipeline for non-goal LIBERO suites.
#
# Example:
#   LIBERO_TASK_SUITE=libero_spatial SKIP_PRETOKENIZE=1 SKIP_ACTION_HIDDEN=1 \
#     bash scripts/prepare_libero_suite_pipeline.sh
#
# Stages:
#   1. raw hdf5 -> no-noop hdf5
#   2. no-noop hdf5 -> pi0.6 remaining reward hdf5
#   3. optional no-noop hdf5 -> expanded image/action/state tree
#   4. optional conversation/token/config generation
#   5. optional RynnVLA action-hidden sidecar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_spatial}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
case "${LIBERO_TASK_SUITE}" in
  libero_goal|libero_object) DEFAULT_ACTION_HORIZON=5 ;;
  *) DEFAULT_ACTION_HORIZON=10 ;;
esac
ACTION_HORIZON="${ACTION_HORIZON:-${DEFAULT_ACTION_HORIZON}}"
TIME_HORIZON="${TIME_HORIZON:-${ACTION_HORIZON}}"
TASK_NAME="${TASK_NAME:-${LIBERO_TASK_SUITE#libero_}}"
LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-${TASK_NAME}}"

RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_ROOT}/data/libero/datasets/${LIBERO_TASK_SUITE}}"
NO_NOOP_DIR="${NO_NOOP_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
REWARD_DIR="${REWARD_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}_pi06_remaining_reward}"
IMAGE_TREE_DIR="${IMAGE_TREE_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_image_state_action_t_${IMAGE_RESOLUTION}}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/data/configs/${LIBERO_TASK_SUITE}}"
HIDDEN_DIR="${HIDDEN_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}_pi0_legacy_action_hidden_vla_policy_h2}"
VLA_INIT_CKPT="${VLA_INIT_CKPT:-${PROJECT_ROOT}/data/ckpts/VLA_model_256/${LIBERO_TASK_SUITE}}"
MODEL_PATH="${MODEL_PATH:-${VLA_INIT_CKPT}}"
ENCODER_STATE_CKPT="${ENCODER_STATE_CKPT:-none}"

SKIP_IMAGE_TREE="${SKIP_IMAGE_TREE:-0}"
SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-1}"
SKIP_ACTION_HIDDEN="${SKIP_ACTION_HIDDEN:-0}"
OVERWRITE="${OVERWRITE:-0}"

echo "=== LIBERO suite pipeline ==="
echo "suite:          ${LIBERO_TASK_SUITE}"
echo "task_name:      ${TASK_NAME}"
echo "resolution:     ${IMAGE_RESOLUTION}"
echo "horizon:        ${ACTION_HORIZON}"
echo "raw:            ${RAW_DATA_DIR}"
echo "no_noop:        ${NO_NOOP_DIR}"
echo "reward:         ${REWARD_DIR}"
echo "image_tree:     ${IMAGE_TREE_DIR}"
echo "configs:        ${CONFIG_DIR}"
echo "hidden:         ${HIDDEN_DIR}"
echo "vla:            ${VLA_INIT_CKPT}"
echo "skip_image:     ${SKIP_IMAGE_TREE}"
echo "skip_pretok:    ${SKIP_PRETOKENIZE}"
echo "skip_hidden:    ${SKIP_ACTION_HIDDEN}"

if [[ ! -d "${RAW_DATA_DIR}" ]]; then
  echo "ERROR: missing raw dataset directory: ${RAW_DATA_DIR}" >&2
  exit 2
fi
if [[ "$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' | wc -l)" -le 0 ]]; then
  echo "ERROR: no *_demo.hdf5 files found under ${RAW_DATA_DIR}" >&2
  exit 2
fi

if [[ ! -d "${NO_NOOP_DIR}" || "${OVERWRITE}" == "1" ]]; then
  echo "=== Stage 1: no-op filtering ==="
  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  RAW_DATA_DIR="${RAW_DATA_DIR}" \
  TARGET_DIR="${NO_NOOP_DIR}" \
  bash scripts/preprocess/processed_data_no_op.sh
else
  echo "=== Stage 1 skipped: ${NO_NOOP_DIR} exists ==="
fi

if [[ ! -d "${REWARD_DIR}" || "${OVERWRITE}" == "1" ]]; then
  echo "=== Stage 2: pi0.6 remaining reward hdf5 ==="
  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  INPUT_DIR="${NO_NOOP_DIR}" \
  OUTPUT_DIR="${REWARD_DIR}" \
  OVERWRITE="${OVERWRITE}" \
  bash scripts/preprocess/processed_data_remaining_steps_reward.sh
else
  echo "=== Stage 2 skipped: ${REWARD_DIR} exists ==="
fi

if [[ "${SKIP_IMAGE_TREE}" != "1" ]]; then
  if [[ ! -d "${IMAGE_TREE_DIR}" || "${OVERWRITE}" == "1" ]]; then
    echo "=== Stage 3: expanded image/action/state tree ==="
    LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
    IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
    RAW_DATA_DIR="${NO_NOOP_DIR}" \
    SAVE_DIR="${IMAGE_TREE_DIR}" \
    bash scripts/preprocess/processed_data_save_img_action_state_wrist.sh
  else
    echo "=== Stage 3 skipped: ${IMAGE_TREE_DIR} exists ==="
  fi
fi

if [[ "${SKIP_PRETOKENIZE}" != "1" ]]; then
  echo "=== Stage 4: convs/tokens/configs ==="
  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  LIBERO_TASK_NAME="${LIBERO_TASK_NAME}" \
  TASK_NAME="${TASK_NAME}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  BASE_DIR="${IMAGE_TREE_DIR}" \
  bash scripts/preprocess/processed_data_generate_convs.sh

  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  TASK_NAME="${TASK_NAME}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  bash scripts/preprocess/processed_data_pretokenize.sh

  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  TASK_NAME="${TASK_NAME}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  CONFIG_DIR="${CONFIG_DIR}" \
  bash scripts/preprocess/prepare_train_configs.sh
else
  echo "=== Stage 4 skipped: pretokenize disabled ==="
  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  TASK_NAME="${TASK_NAME}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  CONFIG_DIR="${CONFIG_DIR}" \
  bash scripts/preprocess/prepare_train_configs.sh
fi

if [[ "${SKIP_ACTION_HIDDEN}" != "1" ]]; then
  if [[ ! -d "${VLA_INIT_CKPT}" ]]; then
    echo "ERROR: missing VLA checkpoint for action-hidden generation: ${VLA_INIT_CKPT}" >&2
    echo "Hint: LIBERO_SUITES=${LIBERO_TASK_SUITE} DOWNLOAD_ACTION_WM=0 bash scripts/download_hf.sh" >&2
    exit 3
  fi
  echo "=== Stage 5: legacy full action-hidden sidecar ==="
  LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  TIME_HORIZON="${TIME_HORIZON}" \
  VLA_INIT_CKPT="${VLA_INIT_CKPT}" \
  MODEL_PATH="${MODEL_PATH}" \
  ENCODER_STATE_CKPT="${ENCODER_STATE_CKPT}" \
  HDF5_DIR="${NO_NOOP_DIR}" \
  OUT_DIR="${HIDDEN_DIR}" \
  ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-legacy}" \
  OBS_HIDDEN_SOURCE="${OBS_HIDDEN_SOURCE:-action_query}" \
  PROMPT_STYLE="${PROMPT_STYLE:-vla_policy}" \
  HISTORY="${HISTORY:-2}" \
  INCLUDE_STATE="${INCLUDE_STATE:-1}" \
  ROTATE_IMAGES_180="${ROTATE_IMAGES_180:-1}" \
  SAVE_ACTION_HIDDEN="${SAVE_ACTION_HIDDEN:-1}" \
  SAVE_ACTOR_SEQUENCE="${SAVE_ACTOR_SEQUENCE:-0}" \
  bash scripts/preprocess_rynn_pixel_hidden.sh
fi

echo "=== LIBERO suite pipeline complete ==="
