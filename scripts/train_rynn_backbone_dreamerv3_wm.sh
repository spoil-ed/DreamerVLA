#!/usr/bin/env bash
# Strategy 3 WM: pixel observations encoded by frozen RynnVLA-002 backbone,
# then trained with the same DreamerV3 RSSM structure as pixel/token DreamerV3.
#
# Defaults are set for the current two-stage plan:
#   1. train the Rynn-hidden precomputed DreamerV3 WM on GPUs 0,1,2,3
#   2. later load its ckpt explicitly into DreamerVLA actor/critic training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env_libero_goal.sh"

CONFIG_NAME="${CONFIG_NAME:-rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed}"
WM_KIND="${WM_KIND:-rynn_backbone}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29513}"
RYNN_PIXEL_DDP="${RYNN_PIXEL_DDP:-1}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
DATALOADER_MP_CONTEXT="${DATALOADER_MP_CONTEXT:-forkserver}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-${DREAMERVLA_UNIFIED_VLA_TAG}_fullseq_ddp_precomputed_bs${BATCH_SIZE}_nw${NUM_WORKERS}_gpu${CUDA_VISIBLE_DEVICES//,/}_viz}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/worldmodel/rynn_backbone_dreamerv3_wm}"
RYNN_HIDDEN_DIR="${RYNN_WM_HIDDEN_DIR:-${RYNN_HIDDEN_FULLSEQ_DIR}}"
LOAD_ACTOR_SEQUENCE="${LOAD_ACTOR_SEQUENCE:-true}"
ACTOR_SEQUENCE_LENGTH="${ACTOR_SEQUENCE_LENGTH:-640}"
FULL_HIDDEN_REC_SCALE="${FULL_HIDDEN_REC_SCALE:-10.0}"

export CONFIG_NAME WM_KIND CUDA_VISIBLE_DEVICES NUM_GPUS MASTER_PORT RYNN_PIXEL_DDP
export BATCH_SIZE NUM_WORKERS PREFETCH_FACTOR PIN_MEMORY PERSISTENT_WORKERS DATALOADER_MP_CONTEXT
export TIMESTAMP RUN_TAG OUT_DIR_BASE
export VLA_INIT_CKPT VLA_STATE_CKPT ENCODER_STATE_CKPT RYNN_HIDDEN_DIR ACTION_HORIZON TIME_HORIZON
export LOAD_ACTOR_SEQUENCE ACTOR_SEQUENCE_LENGTH FULL_HIDDEN_REC_SCALE

if [[ "${DETACH:-0}" == "1" && "${RYNN_WM_DETACHED:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  PREVIEW="$(DRY_RUN=1 "${SCRIPT_DIR}/train_wm.sh" "$@")"
  DETACHED_OUT_DIR="$(printf '%s\n' "${PREVIEW}" | awk -F': ' '/^Run output dir:/ {print $2; exit}')"
  if [[ -z "${DETACHED_OUT_DIR}" ]]; then
    echo "Could not infer output dir from scripts/train_wm.sh dry run." >&2
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

  echo "Launching detached Rynn-pixel precomputed WM training."
  echo "Run output dir: ${DETACHED_OUT_DIR}"
  echo "Stdout log: ${DETACHED_LOG_LABEL}"

  if [[ -n "${DETACHED_LOG}" ]]; then
    DETACH=0 RYNN_WM_DETACHED=1 setsid bash "$0" "$@" > "${DETACHED_LOG}" 2>&1 < /dev/null &
  else
    DETACH=0 RYNN_WM_DETACHED=1 setsid bash "$0" "$@" < /dev/null &
  fi
  DETACHED_PID="$!"
  echo "${DETACHED_PID}" > "${DETACHED_OUT_DIR}/train.pid"
  echo "PID: ${DETACHED_PID}"
  exit 0
fi

exec "${SCRIPT_DIR}/train_wm.sh" "$@"
