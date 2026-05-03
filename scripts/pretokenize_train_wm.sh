#!/usr/bin/env bash
# World-model (TSSM) training.
#
# Usage:
#   conda activate wmpo
#   bash scripts/pretokenize_train_wm.sh
#
# Override via env vars:
#   NUM_GPUS=8 CONFIG_NAME=pretokenize_wm_libero_10 bash scripts/pretokenize_train_wm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_wm_libero_10_transdreamer}"
WM_T="${WM_T:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/pretokenize_wm}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"
CONFIG_FILE="${PROJECT_ROOT}/configs/${CONFIG_NAME}.yaml"
SEQUENCE_OVERRIDES=()

if [[ -n "${WM_T}" && -f "${CONFIG_FILE}" ]] && grep -q "batch_length:" "${CONFIG_FILE}"; then
  SEQUENCE_OVERRIDES=(
    "++dataset.sequence_length=${WM_T}"
    "++dataset_val_ind.sequence_length=${WM_T}"
    "++dataset_val_ood.sequence_length=${WM_T}"
  )
fi

echo "Run output dir: ${OUT_DIR}"
if ((${#SEQUENCE_OVERRIDES[@]})); then
  echo "WM sequence T: ${WM_T}"
else
  echo "WM sequence T: config-controlled (set WM_T to override sequence_length)"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "${SEQUENCE_OVERRIDES[@]}" \
  "$@"
