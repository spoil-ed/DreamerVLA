#!/usr/bin/env bash
# Train the DreamerVLA actor/critic path on top of the precomputed Rynn-pixel
# DreamerV3 world model.
#
# This uses:
#   pixel RGB -> reconstruction target
#   precomputed Rynn hidden -> RSSM observation encoder input
#   Dreamer RSSM [h,z] -> e_hat 4096 -> reused VLA ActionHead actor
#   LIBERO sparse reward -> binary [0,1] reward head
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="${DREAMERVLA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}:$PATH"
export CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_rynn_pixel_precomputed_vlaactor}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-false}"
DROP_LAST="${DROP_LAST:-true}"
DATALOADER_MP_CONTEXT="${DATALOADER_MP_CONTEXT:-forkserver}"
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
else
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
fi
export BATCH_SIZE NUM_WORKERS PREFETCH_FACTOR PIN_MEMORY DROP_LAST PERSISTENT_WORKERS DATALOADER_MP_CONTEXT
export RUN_TAG="${RUN_TAG:-rynn_pixel_ehat4096_vlahead_binary_reward_bs${BATCH_SIZE}_nw${NUM_WORKERS}}"
export OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/dreamervla}"
export PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATALOADER_OVERRIDES=(
  "dataloader.batch_size=${BATCH_SIZE}"
  "dataloader.num_workers=${NUM_WORKERS}"
  "dataloader.pin_memory=${PIN_MEMORY}"
  "dataloader.drop_last=${DROP_LAST}"
  "dataloader.persistent_workers=${PERSISTENT_WORKERS}"
)
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  DATALOADER_OVERRIDES+=("dataloader.prefetch_factor=${PREFETCH_FACTOR}")
  DATALOADER_OVERRIDES+=("dataloader.multiprocessing_context=${DATALOADER_MP_CONTEXT}")
fi

INIT_OVERRIDES=()
if [[ -n "${WORLD_MODEL_STATE_CKPT:-}" ]]; then
  INIT_OVERRIDES+=("init.reset_world_model_reward_head=${RESET_WORLD_MODEL_REWARD_HEAD:-false}")
elif [[ -n "${RESET_WORLD_MODEL_REWARD_HEAD:-}" ]]; then
  INIT_OVERRIDES+=("init.reset_world_model_reward_head=${RESET_WORLD_MODEL_REWARD_HEAD}")
fi

if [[ "${DETACH:-0}" == "1" && "${DREAMERVLA_DETACHED:-0}" != "1" ]]; then
  export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

  PREVIEW="$(DRY_RUN=1 bash scripts/train_dreamer_vla.sh "${DATALOADER_OVERRIDES[@]}" "${INIT_OVERRIDES[@]}" "$@")"
  DETACHED_OUT_DIR="$(printf '%s\n' "${PREVIEW}" | awk -F': ' '/^Run output dir:/ {print $2; exit}')"
  if [[ -z "${DETACHED_OUT_DIR}" ]]; then
    echo "Could not infer output dir from scripts/train_dreamer_vla.sh dry run." >&2
    printf '%s\n' "${PREVIEW}" >&2
    exit 1
  fi

  mkdir -p "${DETACHED_OUT_DIR}"
  if [[ -n "${LOG:-}" ]]; then
    DETACHED_LOG="${LOG}"
    DETACHED_LOG_LABEL="${DETACHED_LOG}"
  else
    DETACHED_LOG=""
    DETACHED_LOG_LABEL="terminal"
  fi

  echo "Launching detached Rynn-pixel DreamerVLA training."
  echo "Run output dir: ${DETACHED_OUT_DIR}"
  echo "Stdout log: ${DETACHED_LOG_LABEL}"

  if [[ -n "${DETACHED_LOG}" ]]; then
    DETACH=0 DREAMERVLA_DETACHED=1 setsid bash "$0" "$@" > "${DETACHED_LOG}" 2>&1 < /dev/null &
  else
    DETACH=0 DREAMERVLA_DETACHED=1 setsid bash "$0" "$@" < /dev/null &
  fi
  DETACHED_PID="$!"
  echo "${DETACHED_PID}" > "${DETACHED_OUT_DIR}/train.pid"
  echo "PID: ${DETACHED_PID}"
  exit 0
fi

exec bash scripts/train_dreamer_vla.sh "${DATALOADER_OVERRIDES[@]}" "${INIT_OVERRIDES[@]}" "$@"
