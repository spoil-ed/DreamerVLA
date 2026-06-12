#!/usr/bin/env bash
# Extract the RynnVLA Scheme-B input-token sidecar used by frame-level WM routes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_TOKEN_HIDDEN_DIR="${INPUT_TOKEN_HIDDEN_DIR:-${HDF5_DIR}_pi0_input_token_embedding_vla_policy_h2}"

if ! has_regular_files "${VLA_CKPT}"; then
  echo "Missing VLA checkpoint files for input-token sidecar: ${VLA_CKPT}" >&2
  echo "Run: LIBERO_SUITES=${TASK} DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
  exit 4
fi
if ! has_regular_files "${TOKENIZER_PATH}"; then
  echo "Missing Lumina tokenizer/backbone files for preprocessing: ${TOKENIZER_PATH}" >&2
  echo "Run: DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
  exit 4
fi
require_file "${TEXT_TOKENIZER_PATH}" "Missing Chameleon text tokenizer for preprocessing" 4
require_file "${CHAMELEON_VQGAN_CONFIG}" "Missing Chameleon VQGAN config for preprocessing" 4
require_file "${CHAMELEON_VQGAN_CKPT}" "Missing Chameleon VQGAN checkpoint for preprocessing" 4
require_hdf5_files "${REWARD_DIR}" "[preprocess:32_input_token_hidden.sh] missing reward HDF5 input" 5

if [[ ! -d "${INPUT_TOKEN_HIDDEN_DIR}" || "${OVERWRITE}" == "1" ]]; then
  preprocess_log "stage 32: RynnVLA input-token sidecar"
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${INPUT_TOKEN_HIDDEN_DIR}"
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${ACTION_HIDDEN_GPUS}" \
    --module dreamer_vla.preprocess.preprocess_rynn_pixel_hidden \
    --hdf5-dir "${REWARD_DIR}" \
    --out-dir "${INPUT_TOKEN_HIDDEN_DIR}" \
    --model-path "${VLA_CKPT}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --text-tokenizer-path "${TEXT_TOKENIZER_PATH}" \
    --chameleon-vqgan-config "${CHAMELEON_VQGAN_CONFIG}" \
    --chameleon-vqgan-ckpt "${CHAMELEON_VQGAN_CKPT}" \
    --action-head-type legacy \
    --obs-hidden-source input_token_embedding \
    --history 2 \
    --include-state \
    --rotate-images-180 \
    --action-dim 7 \
    --time-horizon "${TIME_HORIZON}" \
    --overwrite
else
  preprocess_log "stage 32 skipped: ${INPUT_TOKEN_HIDDEN_DIR}"
fi

require_hdf5_files "${INPUT_TOKEN_HIDDEN_DIR}" "[preprocess:32_input_token_hidden.sh] stage 32 did not create input-token HDF5 files"
