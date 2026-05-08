#!/usr/bin/env bash
# Precompute frozen RynnVLA hidden vectors for LIBERO pixel HDF5 files.
#
# The source pixel dataset is not modified.  Matching sidecar HDF5 files are
# written under data/processed_data/libero_goal_no_noops_t_256_rynn_hidden by
# default, with the same filenames as the source HDF5 files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"

HDF5_DIR="${HDF5_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256_rynn_hidden}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
OUTPUT_DTYPE="${OUTPUT_DTYPE:-float16}"
COMPRESSION="${COMPRESSION:-none}"
RYNN_HIDDEN_RUN_ID="${RYNN_HIDDEN_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ARGS=(
  "${PROJECT_ROOT}/scripts/preprocess_rynn_pixel_hidden.py"
  "--hdf5-dir" "${HDF5_DIR}"
  "--out-dir" "${OUT_DIR}"
  "--chunk-size" "${CHUNK_SIZE}"
  "--output-dtype" "${OUTPUT_DTYPE}"
  "--compression" "${COMPRESSION}"
)

if [[ -n "${MAX_FILES:-}" ]]; then
  ARGS+=("--max-files" "${MAX_FILES}")
fi
if [[ -n "${MAX_DEMOS_PER_FILE:-}" ]]; then
  ARGS+=("--max-demos-per-file" "${MAX_DEMOS_PER_FILE}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=("--overwrite")
fi

echo "[rynn-hidden] source: ${HDF5_DIR}"
echo "[rynn-hidden] output: ${OUT_DIR}"
echo "[rynn-hidden] GPUs:   ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NUM_GPUS})"
echo "[rynn-hidden] run id: ${RYNN_HIDDEN_RUN_ID}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export RYNN_HIDDEN_RUN_ID

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${ARGS[@]}"
fi
