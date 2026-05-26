#!/usr/bin/env bash
# Wait for a LIBERO suite's core preprocessing and VLA checkpoint, then run
# legacy full action-hidden sidecar generation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:?set LIBERO_TASK_SUITE, e.g. libero_spatial}"
GPU="${GPU:-7}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
case "${LIBERO_TASK_SUITE}" in
  libero_goal|libero_object) ACTION_HORIZON="${ACTION_HORIZON:-5}" ;;
  *) ACTION_HORIZON="${ACTION_HORIZON:-10}" ;;
esac
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_ROOT}/data/libero/datasets/${LIBERO_TASK_SUITE}}"
NO_NOOP_DIR="${NO_NOOP_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
REWARD_DIR="${REWARD_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}_pi06_remaining_reward}"
HIDDEN_DIR="${HIDDEN_DIR:-${PROJECT_ROOT}/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}_pi0_legacy_action_hidden_vla_policy_h2}"
VLA_INIT_CKPT="${VLA_INIT_CKPT:-${PROJECT_ROOT}/data/ckpts/VLA_model_256/${LIBERO_TASK_SUITE}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
START_TIME="$(date +%s)"

expected_hdf5_count() {
  find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' | wc -l
}

ready_core() {
  local expected actual_noop actual_reward
  expected="$(expected_hdf5_count)"
  actual_noop="$(find "${NO_NOOP_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' 2>/dev/null | wc -l)"
  actual_reward="$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*_demo.hdf5' 2>/dev/null | wc -l)"
  [[ "${expected}" -gt 0 ]] || return 1
  [[ "${actual_noop}" -eq "${expected}" ]] || return 1
  [[ "${actual_reward}" -eq "${expected}" ]] || return 1
  [[ -f "${REWARD_DIR}/remaining_steps_reward_summary.json" ]] || return 1
}

ready_vla() {
  [[ -f "${VLA_INIT_CKPT}/config.json" ]] || return 1
  [[ -f "${VLA_INIT_CKPT}/model.safetensors.index.json" ]] || return 1
  [[ -f "${VLA_INIT_CKPT}/model-00001-of-00003.safetensors" ]] || return 1
  [[ -f "${VLA_INIT_CKPT}/model-00002-of-00003.safetensors" ]] || return 1
  [[ -f "${VLA_INIT_CKPT}/model-00003-of-00003.safetensors" ]] || return 1
}

while true; do
  if ready_core && ready_vla; then
    break
  fi
  now="$(date +%s)"
  waited="$((now - START_TIME))"
  echo "[hidden-wait] ${LIBERO_TASK_SUITE}: waited=${waited}s core=$(ready_core && echo yes || echo no) vla=$(ready_vla && echo yes || echo no)"
  if [[ "${MAX_WAIT_SECONDS}" -gt 0 && "${waited}" -gt "${MAX_WAIT_SECONDS}" ]]; then
    echo "[hidden-wait] ERROR: timeout waiting for ${LIBERO_TASK_SUITE}" >&2
    exit 2
  fi
  sleep "${POLL_SECONDS}"
done

echo "[hidden-wait] ${LIBERO_TASK_SUITE}: ready, launching hidden generation on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" \
NUM_GPUS=1 \
LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE}" \
IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
ACTION_HORIZON="${ACTION_HORIZON}" \
SKIP_IMAGE_TREE=1 \
SKIP_PRETOKENIZE=1 \
SKIP_ACTION_HIDDEN=0 \
VLA_INIT_CKPT="${VLA_INIT_CKPT}" \
MODEL_PATH="${VLA_INIT_CKPT}" \
ENCODER_STATE_CKPT=none \
HIDDEN_DIR="${HIDDEN_DIR}" \
ACTION_HEAD_TYPE=legacy \
OBS_HIDDEN_SOURCE=action_query \
PROMPT_STYLE=vla_policy \
HISTORY=2 \
INCLUDE_STATE=1 \
ROTATE_IMAGES_180=1 \
SAVE_ACTION_HIDDEN=1 \
SAVE_ACTOR_SEQUENCE=0 \
bash scripts/prepare_libero_suite_pipeline.sh
