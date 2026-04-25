#!/usr/bin/env bash
# World-model (TSSM) training.
#
# Usage:
#   conda activate wmpo
#   bash scripts/train_wm.sh
#
# Override via env vars:
#   NUM_GPUS=8 CONFIG_NAME=pretokenize_wm_libero_10 bash scripts/train_wm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_wm_transdreamer_libero_10_token.yaml}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/pretokenize_wm}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"

echo "Run output dir: ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@"
