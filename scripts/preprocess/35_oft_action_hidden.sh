#!/usr/bin/env bash
# Extract OpenVLA-OFT sidecars consumed by OFT world-model / classifier /
# DreamerVLA routes.
#
# Supports both OFT checkpoint formats (auto-detected by default):
#   - component-wise L1 head (action_head--*_checkpoint.pt present)
#   - merged discrete LM-head (e.g. downloaded one-trajectory weights)
#
# Examples:
#   TASK=libero_goal bash scripts/preprocess/35_oft_action_hidden.sh
#
#   # Scheme B frame-level projected vision tokens:
#   OFT_LATENT_SCHEME=input_tokens TASK=libero_goal bash scripts/preprocess/35_oft_action_hidden.sh
#
#   # Downloaded discrete one-trajectory weights (single view, no history):
#   TASK=libero_goal \
#   OFT_CKPT=data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
#   OFT_POLICY_MODE=discrete OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb \
#   bash scripts/preprocess/35_oft_action_hidden.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

OFT_CKPT="${OFT_CKPT:-${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/${TASK}}"
OFT_POLICY_MODE="${OFT_POLICY_MODE:-auto}"
OFT_LATENT_SCHEME="${OFT_LATENT_SCHEME:-action_hidden}"
OFT_HIDDEN_DIR="${OFT_HIDDEN_DIR:-${HDF5_DIR}_oft_legacy_action_hidden_vla_policy_h2}"
OFT_INPUT_TOKEN_DIR="${OFT_INPUT_TOKEN_DIR:-${HDF5_DIR}_oft_input_token_embedding_vla_policy_h${OFT_HISTORY:-2}}"
OFT_TIME_HORIZON="${OFT_TIME_HORIZON:-8}"
OFT_HISTORY="${OFT_HISTORY:-2}"
OFT_IMAGE_KEYS="${OFT_IMAGE_KEYS:-agentview_rgb eye_in_hand_rgb}"
OFT_ACTION_HIDDEN_GPUS="${OFT_ACTION_HIDDEN_GPUS:-${NGPU:-1}}"
UNNORM_KEY="${UNNORM_KEY:-${TASK}_no_noops}"

if ! has_regular_files "${OFT_CKPT}"; then
  echo "Missing OpenVLA-OFT checkpoint files for action-hidden sidecar: ${OFT_CKPT}" >&2
  echo "Run: DOWNLOAD_OPENVLA_OFT=1 bash scripts/download_assets.sh (or point OFT_CKPT at a local checkpoint)" >&2
  exit 4
fi
require_hdf5_files "${REWARD_DIR}" "[preprocess:35_oft_action_hidden.sh] missing reward HDF5 input" 5

case "${OFT_LATENT_SCHEME}" in
  action_hidden|input_tokens|both) ;;
  *)
    echo "Unsupported OFT_LATENT_SCHEME=${OFT_LATENT_SCHEME}; use action_hidden, input_tokens, or both." >&2
    exit 2
    ;;
esac

need_run=0
if [[ "${OFT_LATENT_SCHEME}" == "action_hidden" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
  [[ ! -d "${OFT_HIDDEN_DIR}" || "${OVERWRITE}" == "1" ]] && need_run=1
fi
if [[ "${OFT_LATENT_SCHEME}" == "input_tokens" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
  [[ ! -d "${OFT_INPUT_TOKEN_DIR}" || "${OVERWRITE}" == "1" ]] && need_run=1
fi

if [[ "${need_run}" == "1" ]]; then
  preprocess_log "stage 35: OpenVLA-OFT sidecar scheme=${OFT_LATENT_SCHEME} mode=${OFT_POLICY_MODE}"
  cmd=(
    "${PYTHON}" -m torch.distributed.run
    --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}"
    --module dreamer_vla.preprocess.preprocess_oft_action_hidden
    --hdf5-dir "${REWARD_DIR}"
    --skip-cd-sidecars
    --oft-ckpt "${OFT_CKPT}"
    --policy-mode "${OFT_POLICY_MODE}"
    --unnorm-key "${UNNORM_KEY}"
    --history "${OFT_HISTORY}"
    --time-horizon "${OFT_TIME_HORIZON}"
    --overwrite
  )
  if [[ "${OFT_LATENT_SCHEME}" == "action_hidden" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}"
    cmd+=(--out-action-dir "${OFT_HIDDEN_DIR}")
  fi
  if [[ "${OFT_LATENT_SCHEME}" == "input_tokens" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_INPUT_TOKEN_DIR}"
    cmd+=(--out-input-token-dir "${OFT_INPUT_TOKEN_DIR}")
  fi
  # shellcheck disable=SC2206
  image_key_args=(${OFT_IMAGE_KEYS})
  cmd+=(--image-keys "${image_key_args[@]}")
  "${cmd[@]}"
else
  preprocess_log "stage 35 skipped: scheme=${OFT_LATENT_SCHEME}"
fi

if [[ "${OFT_LATENT_SCHEME}" == "action_hidden" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
  require_hdf5_files "${OFT_HIDDEN_DIR}" "[preprocess:35_oft_action_hidden.sh] stage 35 did not create action-hidden HDF5 files"
fi
if [[ "${OFT_LATENT_SCHEME}" == "input_tokens" || "${OFT_LATENT_SCHEME}" == "both" ]]; then
  require_hdf5_files "${OFT_INPUT_TOKEN_DIR}" "[preprocess:35_oft_action_hidden.sh] stage 35 did not create input-token HDF5 files"
fi
