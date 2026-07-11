#!/usr/bin/env bash
# Extract the OpenVLA-OFT projected hidden-token sidecar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
OFT_CKPT="${OFT_CKPT:-${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1}"
OFT_HISTORY="${OFT_HISTORY:-1}"
OFT_IMAGE_KEYS="${OFT_IMAGE_KEYS:-agentview_rgb}"
OFT_IMAGE_KEYS_LIST="[${OFT_IMAGE_KEYS// /,}]"
OFT_CHUNK_SIZE="${OFT_CHUNK_SIZE:-1}"
OFT_FAKE_COMPONENTS="${OFT_FAKE_COMPONENTS:-0}"
OFT_HIDDEN_TOKEN_GPUS="${OFT_HIDDEN_TOKEN_GPUS:-${NGPU:-1}}"
OVERWRITE="${OVERWRITE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_remaining_reward"
OFT_HIDDEN_TOKEN_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_oft_hidden_token_vla_policy_h${OFT_HISTORY}"
UNNORM_KEY="${UNNORM_KEY:-${LIBERO_SUITE}_no_noops}"

cd "${DVLA_ROOT}"

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  exit 5
fi

if [[ "${OVERWRITE}" != "1" && -d "${OFT_HIDDEN_TOKEN_DIR}" ]]; then
  if python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
    dir="${OFT_HIDDEN_TOKEN_DIR}" \
    reference_dir="${REWARD_DIR}" \
    match_reference_demos=true \
    match_reference_lengths=true \
    require_complete_attr=true \
    require_config=true; then
    echo "[35_oft_hidden_token] skip hidden-token sidecar: ${OFT_HIDDEN_TOKEN_DIR}"
    exit 0
  fi
  echo "[35_oft_hidden_token] repair incomplete hidden-token sidecar: ${OFT_HIDDEN_TOKEN_DIR}" >&2
fi

if [[ "${OFT_FAKE_COMPONENTS}" != "1" ]]; then
  python - <<'PY'
from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

root = ensure_openvla_oft_on_path()
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

print(
    "[35_oft_hidden_token] openvla_oft_root="
    f"{root} action_dim={ACTION_DIM} num_actions_chunk={NUM_ACTIONS_CHUNK}"
)
PY
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(overwrite=true)
fi
FAKE_ARGS=()
if [[ "${OFT_FAKE_COMPONENTS}" == "1" ]]; then
  FAKE_ARGS=(fake_oft_components=true)
fi

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${OFT_HIDDEN_TOKEN_GPUS}" \
  --module dreamervla.preprocess.preprocess_oft_hidden_token \
  hdf5_dir="${REWARD_DIR}" \
  out_hidden_token_dir="${OFT_HIDDEN_TOKEN_DIR}" \
  obs_hidden_source=hidden_token \
  oft_ckpt="${OFT_CKPT}" \
  policy_mode=discrete \
  unnorm_key="${UNNORM_KEY}" \
  history="${OFT_HISTORY}" \
  time_horizon=8 \
  chunk_size="${OFT_CHUNK_SIZE}" \
  image_keys="${OFT_IMAGE_KEYS_LIST}" \
  patches_per_image=256 \
  "${FAKE_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${OFT_HIDDEN_TOKEN_DIR}" \
  reference_dir="${REWARD_DIR}" \
  match_reference_demos=true \
  match_reference_lengths=true \
  require_complete_attr=true \
  require_config=true
