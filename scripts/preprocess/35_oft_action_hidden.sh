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
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
if [[ "${ARTIFACT_NAME}" == "${TASK_NAME}" && "${TASK_NAME}" != "${LIBERO_SUITE}" ]]; then
  ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"
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

_check_openvla_oft_env() {
  python - <<'PY'
from __future__ import annotations

import sys

try:
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    root = ensure_openvla_oft_on_path()
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
except ModuleNotFoundError as exc:
    print(
        "[35_oft_action_hidden] OpenVLA-OFT dependency import failed. "
        "Run this step in the WMPO/OpenVLA-OFT environment or install the "
        "WMPO OpenVLA-OFT dependencies into the active Python environment. "
        f"Missing module: {exc.name}",
        file=sys.stderr,
    )
    raise SystemExit(12) from exc
except Exception as exc:
    print(f"[35_oft_action_hidden] OpenVLA-OFT environment check failed: {exc}", file=sys.stderr)
    raise SystemExit(12) from exc

print(
    "[35_oft_action_hidden] openvla_oft_root="
    f"{root} action_dim={ACTION_DIM} num_actions_chunk={NUM_ACTIONS_CHUNK}"
)
PY
}

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"
REWARD_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_remaining_reward"
OFT_HIDDEN_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h${OFT_HISTORY}"
OFT_INPUT_TOKEN_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256_oft_input_token_embedding_vla_policy_h${OFT_HISTORY}"
UNNORM_KEY="${UNNORM_KEY:-${LIBERO_SUITE}_no_noops}"

if [[ -z "$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No reward HDF5 files found under: ${REWARD_DIR}" >&2
  echo "Run: bash scripts/preprocess/prepare_libero_data.sh task=${TASK} only=[10_hdf5_reward]" >&2
  exit 5
fi

if [[ "${OFT_LATENT_SCHEME}" == "action_hidden" ]]; then
  if [[ "${OVERWRITE}" != "1" && -d "${OFT_HIDDEN_DIR}" ]]; then
    if python -m dreamervla.preprocess.check_artifacts hdf5-dir \
      --dir "${OFT_HIDDEN_DIR}" \
      --reference-dir "${REWARD_DIR}" \
      --require-complete-attr \
      --require-config \
      --required-demo-dataset action_hidden_states; then
      echo "[35_oft_action_hidden] skip action-hidden: ${OFT_HIDDEN_DIR}"
      exit 0
    fi
    echo "[35_oft_action_hidden] resume incomplete action-hidden sidecar: ${OFT_HIDDEN_DIR}" >&2
  fi
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}"
  _check_openvla_oft_env
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
    --module dreamervla.preprocess.preprocess_oft_action_hidden \
    --hdf5-dir "${REWARD_DIR}" \
    --out-action-dir "${OFT_HIDDEN_DIR}" \
    --skip-cd-sidecars \
    --oft-ckpt "${OFT_CKPT}" \
    --policy-mode "${OFT_POLICY_MODE}" \
    --unnorm-key "${UNNORM_KEY}" \
    --history "${OFT_HISTORY}" \
    --time-horizon 8 \
    --image-keys ${OFT_IMAGE_KEYS} \
    "${OVERWRITE_ARGS[@]}"
elif [[ "${OFT_LATENT_SCHEME}" == "input_tokens" ]]; then
  if [[ "${OVERWRITE}" != "1" && -d "${OFT_INPUT_TOKEN_DIR}" ]]; then
    if python -m dreamervla.preprocess.check_artifacts hdf5-dir \
      --dir "${OFT_INPUT_TOKEN_DIR}" \
      --reference-dir "${REWARD_DIR}" \
      --require-complete-attr \
      --require-config \
      --required-demo-dataset obs_embedding; then
      echo "[35_oft_action_hidden] skip input-token sidecar: ${OFT_INPUT_TOKEN_DIR}"
      exit 0
    fi
    echo "[35_oft_action_hidden] resume incomplete input-token sidecar: ${OFT_INPUT_TOKEN_DIR}" >&2
  fi
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_INPUT_TOKEN_DIR}"
  _check_openvla_oft_env
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
    --module dreamervla.preprocess.preprocess_oft_action_hidden \
    --hdf5-dir "${REWARD_DIR}" \
    --out-input-token-dir "${OFT_INPUT_TOKEN_DIR}" \
    --skip-cd-sidecars \
    --oft-ckpt "${OFT_CKPT}" \
    --policy-mode "${OFT_POLICY_MODE}" \
    --unnorm-key "${UNNORM_KEY}" \
    --history "${OFT_HISTORY}" \
    --time-horizon 8 \
    --image-keys ${OFT_IMAGE_KEYS} \
    "${OVERWRITE_ARGS[@]}"
elif [[ "${OFT_LATENT_SCHEME}" == "both" ]]; then
  if [[ "${OVERWRITE}" != "1" && -d "${OFT_HIDDEN_DIR}" && -d "${OFT_INPUT_TOKEN_DIR}" ]]; then
    if python -m dreamervla.preprocess.check_artifacts hdf5-dir \
      --dir "${OFT_HIDDEN_DIR}" \
      --reference-dir "${REWARD_DIR}" \
      --require-complete-attr \
      --require-config \
      --required-demo-dataset action_hidden_states && \
       python -m dreamervla.preprocess.check_artifacts hdf5-dir \
      --dir "${OFT_INPUT_TOKEN_DIR}" \
      --reference-dir "${REWARD_DIR}" \
      --require-complete-attr \
      --require-config \
      --required-demo-dataset obs_embedding; then
      echo "[35_oft_action_hidden] skip OFT sidecars: ${OFT_HIDDEN_DIR} ${OFT_INPUT_TOKEN_DIR}"
      exit 0
    fi
    echo "[35_oft_action_hidden] resume incomplete OFT sidecars: ${OFT_HIDDEN_DIR} ${OFT_INPUT_TOKEN_DIR}" >&2
  fi
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${OFT_HIDDEN_DIR}" "${OFT_INPUT_TOKEN_DIR}"
  _check_openvla_oft_env
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${OFT_ACTION_HIDDEN_GPUS}" \
    --module dreamervla.preprocess.preprocess_oft_action_hidden \
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
    "${OVERWRITE_ARGS[@]}"
else
  echo "Unsupported OFT_LATENT_SCHEME=${OFT_LATENT_SCHEME}; use action_hidden, input_tokens, or both." >&2
  exit 2
fi
