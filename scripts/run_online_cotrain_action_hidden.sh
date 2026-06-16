#!/usr/bin/env bash
# Online cotrain — Scheme 2 (action-hidden WM). Single Hydra call:
#   one-traj VLA -> parallel online rollout (1 env/rank) -> replay
#   -> WM+classifier warmup -> WM+classifier + slow-policy RL cotrain
# Usage:
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_online_cotrain_action_hidden.sh
#   NUM_GPUS=1 WANDB_MODE=disabled bash scripts/run_online_cotrain_action_hidden.sh training.debug=true
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"  # use cwd LIBERO paths, not stale ~/.libero
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${DVLA_ROOT}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
torchrun --standalone --nproc_per_node="${NUM_GPUS:-1}" \
  -m dreamervla.train experiment=online_cotrain_oft_action_hidden "$@"
