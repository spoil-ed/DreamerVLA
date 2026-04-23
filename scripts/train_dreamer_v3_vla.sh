#!/usr/bin/env bash
# DreamerV3-style VLA training: per-batch alternation of WM SFT step and
# twohot/target-critic actor-critic imagination step.
#
# Usage:
#   conda activate dreamervla
#   bash scripts/train_dreamer_v3_vla.sh
#
# Override via env vars:
#   NUM_GPUS=8 CONFIG_NAME=dreamer_v3_vla_libero_10 bash scripts/train_dreamer_v3_vla.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONFIG_NAME="${CONFIG_NAME:-dreamer_v3_vla_libero_10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/dreamer_v3_vla}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"

echo "Run output dir: ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@"
