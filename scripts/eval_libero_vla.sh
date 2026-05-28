#!/usr/bin/env bash
# ============================================================================
# LIBERO rollout evaluation for VLA and Dreamer checkpoints.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-eval_libero_vla}"
PYTHON="${PYTHON:-python}"

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[eval_libero_vla] config=${CONFIG}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[eval_libero_vla] extra hydra args: $*"

exec "${PYTHON}" -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
