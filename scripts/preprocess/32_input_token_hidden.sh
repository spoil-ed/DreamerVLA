#!/usr/bin/env bash
# Extract the RynnVLA Scheme-B input-token sidecar used by frame-level WM routes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
if [[ "${LIBERO_SUITE}" == "${TASK}" ]]; then
  case "${TASK_NAME}" in
    RynnVLA_LIBERO|OpenVLA_Onetraj_LIBERO) LIBERO_SUITE="libero_goal" ;;
  esac
fi
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
if [[ "${ARTIFACT_NAME}" == "${TASK_NAME}" && "${TASK_NAME}" != "${LIBERO_SUITE}" ]]; then
  ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"
fi
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}"
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_remaining_reward"
INPUT_TOKEN_HIDDEN_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_input_token_embedding_vla_policy_h2"
VLA_CKPT="${VLA_CKPT:-${DVLA_DATA_ROOT}/checkpoints/VLA_model_256/${LIBERO_SUITE}}"
TOKENIZER_PATH="${DVLA_DATA_ROOT}/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768"
TEXT_TOKENIZER_PATH="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/text_tokenizer.json"
CHAMELEON_VQGAN_CONFIG="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.yaml"
CHAMELEON_VQGAN_CKPT="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.ckpt"

TIME_HORIZON=5
if [[ "${LIBERO_SUITE}" == "libero_spatial" || "${LIBERO_SUITE}" == "libero_10" ]]; then
  TIME_HORIZON=10
fi

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  echo "Run: bash scripts/preprocess/prepare_libero_data.sh task=${TASK} only=[10_hdf5_reward]" >&2
  exit 5
fi

if [[ "${OVERWRITE}" != "1" && -d "${INPUT_TOKEN_HIDDEN_DIR}" ]]; then
  if python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
    dir="${INPUT_TOKEN_HIDDEN_DIR}" \
    reference_dir="${REWARD_DIR}" \
    match_reference_demos=true \
    match_reference_lengths=true \
    require_complete_attr=true \
    require_config=true; then
    echo "[32_input_token_hidden] skip input-token sidecar: ${INPUT_TOKEN_HIDDEN_DIR}"
    exit 0
  fi
  echo "[32_input_token_hidden] repair incomplete input-token sidecar: ${INPUT_TOKEN_HIDDEN_DIR}" >&2
fi
if [[ "${OVERWRITE}" == "1" || ! -d "${INPUT_TOKEN_HIDDEN_DIR}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${INPUT_TOKEN_HIDDEN_DIR}"
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${ACTION_HIDDEN_GPUS}" \
    --module dreamervla.preprocess.preprocess_rynn_pixel_hidden \
    hdf5_dir="${REWARD_DIR}" \
    out_dir="${INPUT_TOKEN_HIDDEN_DIR}" \
    model_path="${VLA_CKPT}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    text_tokenizer_path="${TEXT_TOKENIZER_PATH}" \
    chameleon_vqgan_config="${CHAMELEON_VQGAN_CONFIG}" \
    chameleon_vqgan_ckpt="${CHAMELEON_VQGAN_CKPT}" \
    action_head_type=legacy \
    obs_hidden_source=input_token_embedding \
    history=2 \
    include_state=true \
    rotate_images_180=true \
    action_dim=7 \
    time_horizon="${TIME_HORIZON}" \
    overwrite=true
elif [[ "${OVERWRITE}" != "1" ]]; then
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${ACTION_HIDDEN_GPUS}" \
    --module dreamervla.preprocess.preprocess_rynn_pixel_hidden \
    hdf5_dir="${REWARD_DIR}" \
    out_dir="${INPUT_TOKEN_HIDDEN_DIR}" \
    model_path="${VLA_CKPT}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    text_tokenizer_path="${TEXT_TOKENIZER_PATH}" \
    chameleon_vqgan_config="${CHAMELEON_VQGAN_CONFIG}" \
    chameleon_vqgan_ckpt="${CHAMELEON_VQGAN_CKPT}" \
    action_head_type=legacy \
    obs_hidden_source=input_token_embedding \
    history=2 \
    include_state=true \
    rotate_images_180=true \
    action_dim=7 \
    time_horizon="${TIME_HORIZON}"
fi

python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${INPUT_TOKEN_HIDDEN_DIR}" \
  reference_dir="${REWARD_DIR}" \
  match_reference_demos=true \
  match_reference_lengths=true \
  require_complete_attr=true \
  require_config=true
