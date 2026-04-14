#!/usr/bin/env bash
# VLA + World-model co-training (original joint workspace).
#
# Usage:
#   conda activate wmpo
#   bash scripts/train_vla_wm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_sft_wm_vla_smoke}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  "$@"
