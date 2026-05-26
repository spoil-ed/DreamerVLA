#!/usr/bin/env bash
# One-command DreamerVLA data preparation from raw LIBERO HDF5.
#
# Produces all data currently needed by the LIBERO-goal DreamerVLA stack:
#   1. data/processed_data/libero_goal_no_noops_t_256
#   2. data/processed_data/libero_goal_image_state_action_t_256
#   3. data/processed_data/convs, tokens, concate_tokens
#   4. data/configs/libero_goal/*.yaml
#   5. data/processed_data/libero_goal_no_noops_t_256_pi0_action_hidden_vla_policy_h2
#
# Default usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash scripts/prepare_dreamervla_data.sh
#
# Useful overrides:
#   SKIP_PRETOKENIZE=1   only make pixel/no-op HDF5 + action-hidden sidecar
#   SKIP_ACTION_HIDDEN=1 only run the prepare_data five-step pipeline
#   OVERWRITE=1          overwrite incomplete/old Rynn sidecar files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env_libero_goal_pi0_query.sh"

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export ACTION_HORIZON="${ACTION_HORIZON:-5}"
export TASK_NAME="${TASK_NAME:-goal}"
export LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-${TASK_NAME}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export SAVE_ACTOR_SEQUENCE="${SAVE_ACTOR_SEQUENCE:-0}"
export OUT_DIR="${OUT_DIR:-${PI0_ACTION_HIDDEN_DIR}}"
export SKIP_ACTION_HIDDEN="${SKIP_ACTION_HIDDEN:-${SKIP_LATENT:-0}}"

RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_ROOT}/data/libero/datasets/${LIBERO_TASK_SUITE}}"
VLA_INIT_CKPT="${VLA_INIT_CKPT:-${PROJECT_ROOT}/data/ckpts/VLA_model_256/${LIBERO_TASK_SUITE}}"

echo "=== DreamerVLA Data Preparation ==="
echo "Project:      ${PROJECT_ROOT}"
echo "Suite:        ${LIBERO_TASK_SUITE}"
echo "Resolution:   ${IMAGE_RESOLUTION}"
echo "Horizon:      ${ACTION_HORIZON}"
echo "Raw HDF5:     ${RAW_DATA_DIR}"
echo "VLA init:     ${VLA_INIT_CKPT}"
echo "Action hidden out: ${OUT_DIR}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NUM_GPUS})"

if [[ ! -d "${RAW_DATA_DIR}" ]]; then
  echo "ERROR: raw LIBERO HDF5 directory does not exist: ${RAW_DATA_DIR}" >&2
  echo "Expected files like: ${RAW_DATA_DIR}/*_demo.hdf5" >&2
  exit 2
fi
if [[ "$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' | wc -l)" -le 0 ]]; then
  echo "ERROR: no *_demo.hdf5 files found under: ${RAW_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -d "${VLA_INIT_CKPT}" ]]; then
  echo "ERROR: VLA init checkpoint directory does not exist: ${VLA_INIT_CKPT}" >&2
  echo "Hint: LIBERO_SUITES=${LIBERO_TASK_SUITE} bash scripts/download_hf.sh" >&2
  exit 2
fi

if [[ "${SKIP_PRETOKENIZE:-0}" == "1" ]]; then
  echo "=== Pretokenize pipeline skipped ==="
  export HDF5_DIR="${HDF5_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
  if [[ ! -d "${HDF5_DIR}" ]]; then
    echo "ERROR: SKIP_PRETOKENIZE=1 but no no-op filtered HDF5 dir exists: ${HDF5_DIR}" >&2
    exit 2
  fi
  if [[ "${SKIP_ACTION_HIDDEN}" != "1" ]]; then
    bash "${SCRIPT_DIR}/preprocess_rynn_pixel_hidden.sh"
  fi
else
  export PREPARE_LATENT_DATA=0
  bash "${SCRIPT_DIR}/prepare_data.sh"
  if [[ "${SKIP_ACTION_HIDDEN}" != "1" ]]; then
    bash "${SCRIPT_DIR}/preprocess_rynn_pixel_hidden.sh"
  fi
fi

echo "=== DreamerVLA data ready ==="
echo "No-op pixel HDF5: ${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}"
echo "VLA configs:      ${PROJECT_ROOT}/data/configs/${LIBERO_TASK_SUITE}"
echo "Action hidden:    ${OUT_DIR}"
