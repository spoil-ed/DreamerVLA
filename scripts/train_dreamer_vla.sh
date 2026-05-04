#!/usr/bin/env bash
# DreamerVLA training: per-batch alternation of WM SFT step and
# DreamerV3-style twohot/target-critic actor-critic imagination step.
#
# Usage:
#   conda activate dreamervla
#   bash scripts/train_dreamer_vla.sh
#
# Override via env vars:
#   NUM_GPUS=8 CONFIG_NAME=dreamer_vla_libero_10 bash scripts/train_dreamer_vla.sh
#   PYTHON=/path/to/python NUM_GPUS=4 CONFIG_NAME=dreamer_vla_libero_10 bash scripts/train_dreamer_vla.sh
#
# Default run naming:
#   ${CONFIG_NAME}_${RUN_TAG}_${GPU_TAG}_${IMAGE_TAG}_${ACTOR_LOSS_TAG}_${TIMESTAMP}
#
# Examples:
#   dreamer_vla_libero_10_transdreamer_vlaactor_gpu0123_noimg_dreamerv3pg_20260427_145500
#   dreamer_vla_libero_10_transdreamer_vlaactor_ablation1_gpu4567_img_pathwise_20260427_145500
#
# You can still override OUT_DIR directly for exact paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON:-python}"

if [[ -z "${OUT_DIR_BASE:-}" ]]; then
  if [[ "${CONFIG_NAME}" == *"vlaactor"* ]]; then
    OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/dreamervla/dreamer_vla_vlaactor"
  else
    OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/dreamervla"
  fi
fi

CONFIG_PATH="${PROJECT_ROOT}/configs/${CONFIG_NAME}.yaml"

sanitize_tag() {
  local value="$1"
  value="${value//,/_}"
  value="${value// /_}"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value//./}"
  echo "${value}"
}

if [[ -z "${GPU_TAG:-}" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPU_TAG="gpu${CUDA_VISIBLE_DEVICES//,/}"
  else
    GPU_TAG="gpu$(sanitize_tag "${CUDA_VISIBLE_DEVICES}")"
  fi
fi

if [[ -f "${CONFIG_PATH}" ]]; then
  CFG_IMAGE_LOSS="$(awk -F: '/^[[:space:]]*image_loss_enabled:/ {gsub(/[[:space:]]/, "", $2); print $2}' "${CONFIG_PATH}" | tail -n 1)"
  CFG_ACTOR_LOSS="$(awk -F: '/^[[:space:]]*actor_loss_type:/ {gsub(/[[:space:]]/, "", $2); print $2}' "${CONFIG_PATH}" | tail -n 1)"
else
  CFG_IMAGE_LOSS=""
  CFG_ACTOR_LOSS=""
fi

case "${IMAGE_TAG:-}" in
  "") case "${CFG_IMAGE_LOSS:-}" in
        true|True|TRUE) IMAGE_TAG="img" ;;
        false|False|FALSE) IMAGE_TAG="noimg" ;;
        *) IMAGE_TAG="imgunknown" ;;
      esac ;;
esac

case "${ACTOR_LOSS_TAG:-}" in
  "") case "${CFG_ACTOR_LOSS:-dreamerv3}" in
        dreamerv3) ACTOR_LOSS_TAG="dreamerv3" ;;
        dreamerv3_pg) ACTOR_LOSS_TAG="dreamerv3pg" ;;
        policy_gradient) ACTOR_LOSS_TAG="pg" ;;
        pg) ACTOR_LOSS_TAG="pg" ;;
        pathwise) ACTOR_LOSS_TAG="pathwise" ;;
        *) ACTOR_LOSS_TAG="$(sanitize_tag "${CFG_ACTOR_LOSS}")" ;;
      esac ;;
esac

RUN_NAME="${RUN_NAME:-${CONFIG_NAME}${RUN_TAG:+_${RUN_TAG}}_${GPU_TAG}_${IMAGE_TAG}_${ACTOR_LOSS_TAG}_${TIMESTAMP}}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${RUN_NAME}}"

echo "Run output dir: ${OUT_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Python: ${PYTHON_BIN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching training."
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_BIN} -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} --module src.cli.train --config-name ${CONFIG_NAME} training.out_dir=${OUT_DIR} $*"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@"
