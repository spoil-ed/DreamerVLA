#!/usr/bin/env bash
# Extract the legacy RynnVLA action-hidden sidecar used by action-level WM routes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}"
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${TASK}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_pi06_remaining_reward"
HIDDEN_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
VLA_CKPT="${VLA_CKPT:-${DVLA_DATA_ROOT}/checkpoints/VLA_model_256/${TASK}}"
TOKENIZER_PATH="${DVLA_DATA_ROOT}/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768"
TEXT_TOKENIZER_PATH="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/text_tokenizer.json"
CHAMELEON_VQGAN_CONFIG="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.yaml"
CHAMELEON_VQGAN_CKPT="${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.ckpt"

TIME_HORIZON=5
if [[ "${TASK}" == "libero_spatial" || "${TASK}" == "libero_10" ]]; then
  TIME_HORIZON=10
fi

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  echo "Run: bash scripts/preprocess/prepare_libero_data.sh task=${TASK} only=[10_hdf5_reward]" >&2
  exit 5
fi

if [[ "${OVERWRITE}" == "1" || ! -d "${HIDDEN_DIR}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${HIDDEN_DIR}"
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${ACTION_HIDDEN_GPUS}" \
    --module dreamer_vla.preprocess.preprocess_rynn_pixel_hidden \
    --hdf5-dir "${REWARD_DIR}" \
    --out-dir "${HIDDEN_DIR}" \
    --model-path "${VLA_CKPT}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --text-tokenizer-path "${TEXT_TOKENIZER_PATH}" \
    --chameleon-vqgan-config "${CHAMELEON_VQGAN_CONFIG}" \
    --chameleon-vqgan-ckpt "${CHAMELEON_VQGAN_CKPT}" \
    --action-head-type legacy \
    --obs-hidden-source action_query \
    --history 2 \
    --include-state \
    --rotate-images-180 \
    --save-action-hidden \
    --action-dim 7 \
    --time-horizon "${TIME_HORIZON}" \
    --overwrite
else
  echo "[30_action_hidden] skip action-hidden: ${HIDDEN_DIR}"
fi
