#!/usr/bin/env bash
# One-click 8-GPU staged full-VLA cotrain from the selected warm WM/CLS states.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

export WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/world_model/20260712_052904/ckpt/warmup_topk/wm/wm_step=00004000-loss=0.097758.ckpt}"
export CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/classifier/20260712_052906/checkpoints/best_window_f10.9711_th0.45.ckpt}"
export COTRAIN_RUN_ROOT="${COTRAIN_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/pre_mainline/wmcls_cotrain_ray_eval/$(date +%Y%m%d_%H%M%S)_staged_full_vla}"
WMCLS_COTRAIN_GLOBAL_STEPS="${WMCLS_COTRAIN_GLOBAL_STEPS:-20000}"

echo "[wmcls-cotrain-oneclick] gpus=${CUDA_VISIBLE_DEVICES}" >&2
echo "[wmcls-cotrain-oneclick] world_model=${WORLD_MODEL_CKPT}" >&2
echo "[wmcls-cotrain-oneclick] classifier=${CLASSIFIER_CKPT}" >&2
echo "[wmcls-cotrain-oneclick] run_root=${COTRAIN_RUN_ROOT}" >&2
echo "[wmcls-cotrain-oneclick] global_steps=${WMCLS_COTRAIN_GLOBAL_STEPS}" >&2

exec bash "${SCRIPT_DIR}/e2e_wmcls_cotrain_eval.sh" \
  manual_cotrain.global_steps="${WMCLS_COTRAIN_GLOBAL_STEPS}" \
  "$@"
