#!/usr/bin/env bash
# Extract OpenVLA-OFT Scheme-A action-hidden and/or Scheme-B input-token sidecars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-libero_goal}"
OFT_CKPT="${OFT_CKPT:-${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/${TASK}}"
OFT_POLICY_MODE="${OFT_POLICY_MODE:-auto}"
OFT_LATENT_SCHEME="${OFT_LATENT_SCHEME:-action_hidden}"
OFT_HISTORY="${OFT_HISTORY:-2}"
OFT_IMAGE_KEYS="${OFT_IMAGE_KEYS:-agentview_rgb eye_in_hand_rgb}"
OFT_ACTION_HIDDEN_GPUS="${OFT_ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --ckpt) OFT_CKPT="$2"; shift 2 ;;
    --policy-mode) OFT_POLICY_MODE="$2"; shift 2 ;;
    --scheme) OFT_LATENT_SCHEME="$2"; shift 2 ;;
    --history) OFT_HISTORY="$2"; shift 2 ;;
    --image-keys) OFT_IMAGE_KEYS="$2"; shift 2 ;;
    --gpus)
      export CUDA_VISIBLE_DEVICES="$2"
      if [[ -z "${OFT_ACTION_HIDDEN_GPUS}" ]]; then
        gpu_count=0
        for _gpu in ${2//,/ }; do gpu_count=$((gpu_count + 1)); done
        OFT_ACTION_HIDDEN_GPUS="${gpu_count}"
      fi
      shift 2
      ;;
    --ngpu) OFT_ACTION_HIDDEN_GPUS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
OFT_ACTION_HIDDEN_GPUS="${OFT_ACTION_HIDDEN_GPUS:-1}"
export PYTHON
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data"
REWARD_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_pi06_remaining_reward"
OFT_HIDDEN_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h${OFT_HISTORY}"
OFT_INPUT_TOKEN_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256_oft_input_token_embedding_vla_policy_h${OFT_HISTORY}"
UNNORM_KEY="${TASK}_no_noops"

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  echo "Run: bash scripts/preprocess/10_hdf5_reward.sh --task ${TASK}" >&2
  exit 5
fi

case "${OFT_LATENT_SCHEME}" in
  action_hidden)
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
      --module dreamer_vla.preprocess.preprocess_oft_action_hidden \
      --hdf5-dir "${REWARD_DIR}" \
      --out-action-dir "${OFT_HIDDEN_DIR}" \
      --skip-cd-sidecars \
      --oft-ckpt "${OFT_CKPT}" \
      --policy-mode "${OFT_POLICY_MODE}" \
      --unnorm-key "${UNNORM_KEY}" \
      --history "${OFT_HISTORY}" \
      --time-horizon 8 \
      --image-keys ${OFT_IMAGE_KEYS} \
      --overwrite
    ;;
  input_tokens)
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_INPUT_TOKEN_DIR}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
      --module dreamer_vla.preprocess.preprocess_oft_action_hidden \
      --hdf5-dir "${REWARD_DIR}" \
      --out-input-token-dir "${OFT_INPUT_TOKEN_DIR}" \
      --skip-cd-sidecars \
      --oft-ckpt "${OFT_CKPT}" \
      --policy-mode "${OFT_POLICY_MODE}" \
      --unnorm-key "${UNNORM_KEY}" \
      --history "${OFT_HISTORY}" \
      --time-horizon 8 \
      --image-keys ${OFT_IMAGE_KEYS} \
      --overwrite
    ;;
  both)
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}" "${OFT_INPUT_TOKEN_DIR}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
      --module dreamer_vla.preprocess.preprocess_oft_action_hidden \
      --hdf5-dir "${REWARD_DIR}" \
      --out-action-dir "${OFT_HIDDEN_DIR}" \
      --out-input-token-dir "${OFT_INPUT_TOKEN_DIR}" \
      --skip-cd-sidecars \
      --oft-ckpt "${OFT_CKPT}" \
      --policy-mode "${OFT_POLICY_MODE}" \
      --unnorm-key "${UNNORM_KEY}" \
      --history "${OFT_HISTORY}" \
      --time-horizon 8 \
      --image-keys ${OFT_IMAGE_KEYS} \
      --overwrite
    ;;
  *)
    echo "Unsupported OFT_LATENT_SCHEME=${OFT_LATENT_SCHEME}; use action_hidden, input_tokens, or both." >&2
    exit 2
    ;;
esac
