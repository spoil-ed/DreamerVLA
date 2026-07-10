#!/usr/bin/env bash
# Prepare the complete LIBERO replay required by full-dataset WM training.
# Every stage is resumable; set PREPROCESS_OVERWRITE=true to rebuild it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
echo "[wm-full-dataset-prepare] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-full-dataset-prepare] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT}" >&2

# WM consumes input-token sidecars. Keeping this default narrow avoids building
# the separate action-hidden sidecar unless the caller explicitly requests it.
export PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-false}"
export PREPROCESS_ONLY="${PREPROCESS_ONLY:-[10_hdf5_reward,35_oft_action_hidden]}"
export OFT_LATENT_SCHEME="${OFT_LATENT_SCHEME:-input_tokens}"
export PYTHON_BIN="${PYTHON:-python}"
export PYTHON="${PYTHON_BIN}"
export PREPROCESS_NUM_PROCS="${PREPROCESS_NUM_PROCS:-32}"
export PREPROCESS_GPUS="${PREPROCESS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PREPROCESS_NGPU="${PREPROCESS_NGPU:-${NGPU:-}}"

exec "${DVLA_ROOT}/scripts/experiments/libero_original_00_reprocess_data.sh" "$@"
