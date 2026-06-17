#!/usr/bin/env bash
# Cold-start parallel rollout collection (pure-Hydra entry).
#   torchrun M ranks (one GPU each) drive a base OFT VLA in LIBERO and dump
#   reward-dir-compatible HDF5 + obs_embedding sidecars that the discrete WM
#   (experiment=oft_discrete_token_world_model_dinowm_chunk) consumes zero-change.
#   Config-first: ckpt / dirs / expected_* all come from task=OpenVLA_Onetraj_ColdStart_LIBERO.
#   See docs/superpowers/specs/2026-06-17-coldstart-collector-hydra-and-wm-consumption.md.
#
# Usage:
#   # 2 GPUs, all libero_goal tasks, 300 episodes each, K=8 envs/GPU:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_collect_rollouts.sh \
#       collect.task_ids=all collect.episodes_per_task=300 \
#       collect.episode_horizon=300 collect.envs_per_gpu=8
#
#   # 1-GPU smoke:
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_rollouts.sh \
#       collect.task_ids=0 collect.episodes_per_task=2 \
#       collect.episode_horizon=64 collect.envs_per_gpu=2
#
# Overrides are Hydra key=value (e.g. collect.*, task=, experiment=).
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
cd "${DVLA_ROOT}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_GPUS="${NUM_GPUS:-2}"
# experiment= default can be overridden by passing your own experiment=... in "$@".
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m dreamervla.train experiment=collect_rollouts_onetraj "$@"
