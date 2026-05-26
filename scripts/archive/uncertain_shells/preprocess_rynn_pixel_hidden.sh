#!/usr/bin/env bash
# Precompute frozen RynnVLA hidden vectors for LIBERO pixel HDF5 files.
#
# The source pixel dataset is not modified.  Matching sidecar HDF5 files are
# written under the canonical LIBERO-goal hidden sidecar directory by default,
# with the same filenames as the source HDF5 files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/env_libero_goal_pi0_query.sh"

PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"

HDF5_DIR="${HDF5_DIR:-${PROJECT_ROOT}/data/processed_data/libero_goal_no_noops_t_256}"
OUT_DIR="${OUT_DIR:-${RYNN_HIDDEN_DIR}}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
OUTPUT_DTYPE="${OUTPUT_DTYPE:-float16}"
COMPRESSION="${COMPRESSION:-none}"
SAVE_ACTOR_SEQUENCE="${SAVE_ACTOR_SEQUENCE:-0}"
OBS_HIDDEN_SOURCE="${OBS_HIDDEN_SOURCE:-${PI0_QUERY_OBS_HIDDEN_SOURCE:-action_query}}"
SAVE_ACTION_HIDDEN="${SAVE_ACTION_HIDDEN:-1}"
ACTION_TRIGGER_TOKEN_ID="${ACTION_TRIGGER_TOKEN_ID:-10004}"
PROMPT_STYLE="${PROMPT_STYLE:-${PI0_QUERY_PROMPT_STYLE:-vla_policy}}"
HISTORY="${HISTORY:-${PI0_QUERY_HISTORY:-2}}"
INCLUDE_STATE="${INCLUDE_STATE:-${PI0_QUERY_INCLUDE_STATE:-1}}"
ROTATE_IMAGES_180="${ROTATE_IMAGES_180:-${PI0_QUERY_ROTATE_IMAGES_180:-1}}"
RYNN_HIDDEN_RUN_ID="${RYNN_HIDDEN_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-${VLA_INIT_CKPT}}"
ENCODER_STATE_CKPT="${ENCODER_STATE_CKPT:-${VLA_STATE_CKPT}}"
TIME_HORIZON="${TIME_HORIZON:-${ACTION_HORIZON}}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-pi0_query}"
if [[ "${ENCODER_STATE_CKPT}" == "none" || "${ENCODER_STATE_CKPT}" == "null" || "${ENCODER_STATE_CKPT}" == "__none__" ]]; then
  ENCODER_STATE_CKPT=""
fi

if [[ "${ACTION_HEAD_TYPE}" == "pi0_query" ]]; then
  if [[ "${OBS_HIDDEN_SOURCE}" != "action_query" || "${PROMPT_STYLE}" != "vla_policy" || "${HISTORY}" != "2" || "${INCLUDE_STATE}" != "1" || "${ROTATE_IMAGES_180}" != "1" ]]; then
    echo "[rynn-hidden] ERROR: pi0_query preprocessing must match existing sidecar: vla_policy + history=2 + state + rotate180 + action_query" >&2
    exit 2
  fi
fi

if [[ "${ACTION_HEAD_TYPE}" == "legacy" ]]; then
  if [[ "${OBS_HIDDEN_SOURCE}" != "action_query" || "${PROMPT_STYLE}" != "vla_policy" || "${HISTORY}" != "2" || "${INCLUDE_STATE}" != "1" || "${ROTATE_IMAGES_180}" != "1" ]]; then
    echo "[rynn-hidden] ERROR: legacy preprocessing must match existing sidecar: vla_policy + history=2 + state + rotate180 + action_query" >&2
    exit 2
  fi
  if [[ "${OUT_DIR}" == *"_pi0_action_hidden_"* ]]; then
    echo "[rynn-hidden] ERROR: ACTION_HEAD_TYPE=legacy is about to write 35840-d hidden into a pi0_query (5120-d) sidecar dir: ${OUT_DIR}" >&2
    echo "[rynn-hidden]        Source env_libero_goal_pi0_query.sh AFTER setting ACTION_HEAD_TYPE=legacy, or set OUT_DIR explicitly." >&2
    exit 2
  fi
fi

ARGS=(
  "${PROJECT_ROOT}/scripts/preprocess_rynn_pixel_hidden.py"
  "--hdf5-dir" "${HDF5_DIR}"
  "--out-dir" "${OUT_DIR}"
  "--chunk-size" "${CHUNK_SIZE}"
  "--output-dtype" "${OUTPUT_DTYPE}"
  "--compression" "${COMPRESSION}"
  "--obs-hidden-source" "${OBS_HIDDEN_SOURCE}"
  "--prompt-style" "${PROMPT_STYLE}"
  "--history" "${HISTORY}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=("--model-path" "${MODEL_PATH}")
fi
if [[ -n "${ENCODER_STATE_CKPT}" ]]; then
  ARGS+=("--encoder-state-ckpt" "${ENCODER_STATE_CKPT}")
fi
if [[ -n "${TIME_HORIZON}" ]]; then
  ARGS+=("--time-horizon" "${TIME_HORIZON}")
fi
ARGS+=("--action-head-type" "${ACTION_HEAD_TYPE}")
if [[ -n "${MAX_FILES:-}" ]]; then
  ARGS+=("--max-files" "${MAX_FILES}")
fi
if [[ -n "${MAX_DEMOS_PER_FILE:-}" ]]; then
  ARGS+=("--max-demos-per-file" "${MAX_DEMOS_PER_FILE}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=("--overwrite")
fi
if [[ "${SAVE_ACTOR_SEQUENCE}" == "1" ]]; then
  ARGS+=("--save-actor-sequence" "--action-trigger-token-id" "${ACTION_TRIGGER_TOKEN_ID}")
fi
if [[ "${SAVE_ACTION_HIDDEN}" == "1" ]]; then
  ARGS+=("--save-action-hidden" "--action-trigger-token-id" "${ACTION_TRIGGER_TOKEN_ID}")
fi
if [[ "${INCLUDE_STATE}" == "1" ]]; then
  ARGS+=("--include-state")
fi
if [[ "${ROTATE_IMAGES_180}" == "1" ]]; then
  ARGS+=("--rotate-images-180")
fi

echo "[rynn-hidden] source: ${HDF5_DIR}"
echo "[rynn-hidden] output: ${OUT_DIR}"
echo "[rynn-hidden] GPUs:   ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NUM_GPUS})"
echo "[rynn-hidden] run id: ${RYNN_HIDDEN_RUN_ID}"
echo "[rynn-hidden] obs hidden source: ${OBS_HIDDEN_SOURCE}"
echo "[rynn-hidden] prompt style: ${PROMPT_STYLE} history=${HISTORY} include_state=${INCLUDE_STATE} rotate180=${ROTATE_IMAGES_180}"
echo "[rynn-hidden] actor sequence: ${SAVE_ACTOR_SEQUENCE}"
echo "[rynn-hidden] action hidden: ${SAVE_ACTION_HIDDEN}"
echo "[rynn-hidden] action head: ${ACTION_HEAD_TYPE}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export RYNN_HIDDEN_RUN_ID

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${ARGS[@]}"
fi
