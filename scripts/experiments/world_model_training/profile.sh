#!/usr/bin/env bash
# One-click 8xH100 timing run for official-data world-model training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPU_COUNT="${GPU_COUNT:-8}"
export WORLD_MODEL_EXPERIMENT="${WORLD_MODEL_EXPERIMENT:-wm_official_upper_bound_profile}"
export WORLD_MODEL_RUN_ROOT="${WORLD_MODEL_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/world_model_profile/$(date +%Y%m%d_%H%M%S)}"
export WORLD_MODEL_CHECKPOINT_EVERY="${WORLD_MODEL_CHECKPOINT_EVERY:-0}"

exec bash "${SCRIPT_DIR}/train.sh" "$@"
