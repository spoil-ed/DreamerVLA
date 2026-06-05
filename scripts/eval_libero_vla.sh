#!/usr/bin/env bash
# ============================================================================
# LIBERO rollout evaluation for VLA and Dreamer checkpoints.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

CONFIG="${CONFIG:-eval_libero_vla}"

echo "[eval_libero_vla] config=${CONFIG}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[eval_libero_vla] extra hydra args: $*"

exec "${PYTHON}" -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
