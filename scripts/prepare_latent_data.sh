#!/usr/bin/env bash
# One-shot latent data preparation pipeline:
# existing LIBERO pixel HDF5 -> precomputed RynnVLA hidden sidecar HDF5.
#
# The source pixel dataset is never overwritten.  By default this writes the
# full action-head sequence sidecar:
#   data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000_fullseq
#
# Usage:
#   bash scripts/prepare_latent_data.sh
#
# Override defaults via env vars:
#   LIBERO_TASK_SUITE=libero_goal IMAGE_RESOLUTION=256 CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     bash scripts/prepare_latent_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env_libero_goal.sh"

export LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export CHUNK_SIZE="${CHUNK_SIZE:-16}"
export OUTPUT_DTYPE="${OUTPUT_DTYPE:-float16}"
export COMPRESSION="${COMPRESSION:-none}"
export SAVE_ACTOR_SEQUENCE="${SAVE_ACTOR_SEQUENCE:-1}"

DEFAULT_HDF5_DIR="${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}"
if [[ "${SAVE_ACTOR_SEQUENCE}" == "1" ]]; then
  DEFAULT_OUT_DIR="${RYNN_HIDDEN_FULLSEQ_DIR}"
else
  DEFAULT_OUT_DIR="${RYNN_HIDDEN_DIR}"
fi

export HDF5_DIR="${HDF5_DIR:-${DEFAULT_HDF5_DIR}}"
export OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"

echo "=== Latent Data Preparation: Rynn hidden sidecar ==="
echo "Project:     ${PROJECT_ROOT}"
echo "Source HDF5: ${HDF5_DIR}"
echo "Output dir:  ${OUT_DIR}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NUM_GPUS})"
echo "Chunk size:  ${CHUNK_SIZE}"
echo "Dtype:       ${OUTPUT_DTYPE}"
echo "Full seq:    ${SAVE_ACTOR_SEQUENCE}"

echo "=== Step 1/3: Check source pixel HDF5 dataset ==="
if [[ ! -d "${HDF5_DIR}" ]]; then
  echo "ERROR: source HDF5 directory does not exist: ${HDF5_DIR}" >&2
  echo "Hint: create it first with: bash scripts/prepare_data.sh" >&2
  exit 2
fi

num_hdf5="$(find "${HDF5_DIR}" -maxdepth 1 -type f -name '*.hdf5' | wc -l)"
if [[ "${num_hdf5}" -le 0 ]]; then
  echo "ERROR: no .hdf5 files found under: ${HDF5_DIR}" >&2
  exit 2
fi
echo "Found ${num_hdf5} source HDF5 file(s)."

echo "=== Step 2/3: Precompute frozen RynnVLA hidden vectors ==="
bash "${SCRIPT_DIR}/preprocess_rynn_pixel_hidden.sh"

echo "=== Step 3/3: Latent sidecar ready ==="
echo "Output: ${OUT_DIR}"
echo
echo "Train with precomputed hidden:"
echo "  CONFIG_NAME=rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed \\"
echo "  RYNN_PIXEL_DDP=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS} \\"
echo "  bash scripts/train_rynn_backbone_dreamerv3_wm.sh"
echo
echo "=== Latent data preparation complete ==="
