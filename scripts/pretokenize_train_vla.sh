#!/usr/bin/env bash
# VLA training with LIBERO rollout evaluation after each epoch.
#
# Usage:
#   conda activate wmpo
#   bash scripts/train_vla.sh
#
# Override via env vars:
#   NUM_GPUS=8 CONFIG_NAME=pretokenize_vla_libero_10 bash scripts/train_vla.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-pretokenize_vla_libero_10}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" scripts/train.py \
  --config-name "${CONFIG_NAME}" \
  "$@"
