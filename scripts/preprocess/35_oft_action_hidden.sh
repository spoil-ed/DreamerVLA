#!/usr/bin/env bash
# Extract the OpenVLA-OFT action-hidden sidecar consumed by the OFT
# world-model / classifier / DreamerVLA routes.
#
# Supports both OFT checkpoint formats (auto-detected by default):
#   - component-wise L1 head (action_head--*_checkpoint.pt present)
#   - merged discrete LM-head (e.g. downloaded one-trajectory weights)
#
# Examples:
#   TASK=libero_goal bash scripts/preprocess/35_oft_action_hidden.sh
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
OFT_HIDDEN_DIR="${OFT_HIDDEN_DIR:-${HDF5_DIR}_oft_legacy_action_hidden_vla_policy_h2}"
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

if [[ ! -d "${OFT_HIDDEN_DIR}" || "${OVERWRITE}" == "1" ]]; then
  preprocess_log "stage 35: OpenVLA-OFT action-hidden sidecar (mode=${OFT_POLICY_MODE})"
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}"
  # shellcheck disable=SC2086
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
    --module dreamer_vla.preprocess.preprocess_oft_action_hidden \
    --hdf5-dir "${REWARD_DIR}" \
    --out-action-dir "${OFT_HIDDEN_DIR}" \
    --skip-cd-sidecars \
    --oft-ckpt "${OFT_CKPT}" \
    --policy-mode "${OFT_POLICY_MODE}" \
    --unnorm-key "${UNNORM_KEY}" \
    --image-keys ${OFT_IMAGE_KEYS} \
    --history "${OFT_HISTORY}" \
    --time-horizon "${OFT_TIME_HORIZON}" \
    --overwrite
else
  preprocess_log "stage 35 skipped: ${OFT_HIDDEN_DIR}"
fi

require_hdf5_files "${OFT_HIDDEN_DIR}" "[preprocess:35_oft_action_hidden.sh] stage 35 did not create action-hidden HDF5 files"
