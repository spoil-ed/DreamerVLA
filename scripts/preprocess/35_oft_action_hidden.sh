#!/usr/bin/env bash
# Extract OpenVLA-OFT Scheme-A action-hidden and/or Scheme-B input-token sidecars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
if [[ "${LIBERO_SUITE}" == "${TASK}" ]]; then
  case "${TASK_NAME}" in
    RynnVLA_LIBERO|OpenVLA_Onetraj_LIBERO) LIBERO_SUITE="libero_goal" ;;
  esac
fi
OFT_CKPT="${OFT_CKPT:-${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/${LIBERO_SUITE}}"
OFT_POLICY_MODE="${OFT_POLICY_MODE:-auto}"
OFT_LATENT_SCHEME="${OFT_LATENT_SCHEME:-action_hidden}"
OFT_HISTORY="${OFT_HISTORY:-2}"
OFT_IMAGE_KEYS="${OFT_IMAGE_KEYS:-agentview_rgb eye_in_hand_rgb}"
OFT_ACTION_HIDDEN_GPUS="${OFT_ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"
OFT_ACTION_HIDDEN_GPUS="${OFT_ACTION_HIDDEN_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}"
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${TASK_NAME}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/${TASK_NAME}_no_noops_t_256_pi06_remaining_reward"
OFT_HIDDEN_DIR="${PROCESSED_DATA_ROOT}/${TASK_NAME}_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h${OFT_HISTORY}"
OFT_INPUT_TOKEN_DIR="${PROCESSED_DATA_ROOT}/${TASK_NAME}_no_noops_t_256_oft_input_token_embedding_vla_policy_h${OFT_HISTORY}"
UNNORM_KEY="${UNNORM_KEY:-${LIBERO_SUITE}_no_noops}"

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  echo "Run: bash scripts/preprocess/prepare_libero_data.sh task=${TASK} only=[10_hdf5_reward]" >&2
  exit 5
fi

if [[ "${OFT_LATENT_SCHEME}" == "action_hidden" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}"
  python -m torch.distributed.run \
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
elif [[ "${OFT_LATENT_SCHEME}" == "input_tokens" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_INPUT_TOKEN_DIR}"
  python -m torch.distributed.run \
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
elif [[ "${OFT_LATENT_SCHEME}" == "both" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}" "${OFT_INPUT_TOKEN_DIR}"
  python -m torch.distributed.run \
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
else
  echo "Unsupported OFT_LATENT_SCHEME=${OFT_LATENT_SCHEME}; use action_hidden, input_tokens, or both." >&2
  exit 2
fi
