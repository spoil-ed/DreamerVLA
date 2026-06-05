#!/usr/bin/env bash
# GPUs 4,5 VLA SFT. Switch task with argv[1] or TAG/TASK.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

TASK="${TASK:-${TAG:-libero_10}}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  TASK="$1"
  shift
fi

case "${TASK}" in libero_10|libero_object|libero_spatial) ;; *) echo "ERROR: TASK must be one of: libero_10, libero_object, libero_spatial" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" GPUS="${GPUS:-4,5}" PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-4,5}"
export NGPU="${NGPU:-2}"
export NUM_GPUS="${NUM_GPUS:-${NGPU}}"
export MASTER_PORT="${MASTER_PORT:-29545}"

CONFIG="${CONFIG:-vla_pi0_query}"
CKPT="${CKPT:-${DVLA_ROOT}/data/ckpts/VLA_model_256/${TASK}}"
RUN_TAG="${RUN_TAG:-${TASK}_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-${DVLA_ROOT}/data/outputs/vla/pi0_query/${RUN_TAG}}"

[[ -f "${CKPT}/config.json" ]] || { echo "ERROR: missing pretrained checkpoint: ${CKPT}/config.json" >&2; exit 3; }
HORIZON="$("${PYTHON}" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("time_horizon") or c.get("action_horizon") or "")' "${CKPT}/config.json")"
[[ -n "${HORIZON}" ]] || { echo "ERROR: missing time_horizon/action_horizon in ${CKPT}/config.json" >&2; exit 4; }
EXTRA=()
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA+=("dataloader.batch_size=${BATCH_SIZE}")

exec bash scripts/train_vla.sh \
  "task=${TASK}" \
  "init.vla_ckpt_path=${CKPT}" \
  "encoder.model_path=${CKPT}" \
  "task.action_horizon=${HORIZON}" \
  "task.time_horizon=${HORIZON}" \
  "encoder.time_horizon=${HORIZON}" \
  "${EXTRA[@]}" \
  "$@"
