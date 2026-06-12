#!/usr/bin/env bash
# Extract the RynnVLA Scheme-B input-token sidecar used by frame-level WM routes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-libero_goal}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --gpus)
      export CUDA_VISIBLE_DEVICES="$2"
      if [[ -z "${ACTION_HIDDEN_GPUS}" ]]; then
        gpu_count=0
        for _gpu in ${2//,/ }; do gpu_count=$((gpu_count + 1)); done
        ACTION_HIDDEN_GPUS="${gpu_count}"
      fi
      shift 2
      ;;
    --ngpu) ACTION_HIDDEN_GPUS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-1}"
export PYTHON
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data"
REWARD_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_pi06_remaining_reward"
INPUT_TOKEN_HIDDEN_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_pi0_input_token_embedding_vla_policy_h2"
VLA_CKPT="${DVLA_DATA_ROOT}/checkpoints/VLA_model_256/${TASK}"
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
  echo "Run: bash scripts/preprocess/10_hdf5_reward.sh --task ${TASK}" >&2
  exit 5
fi

if [[ "${OVERWRITE}" == "1" || ! -d "${INPUT_TOKEN_HIDDEN_DIR}" ]]; then
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
  echo "[32_input_token_hidden] skip input-token sidecar: ${INPUT_TOKEN_HIDDEN_DIR}"
fi
