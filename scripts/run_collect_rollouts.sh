#!/usr/bin/env bash
# Cold-start parallel rollout collection:
#   M ranks (torchrun, one GPU each) x K env subprocesses per rank, with batched VLA
#   inference over the K observations.  Drives a base OFT VLA in LIBERO and dumps
#   reward-dir-compatible HDF5 + obs_embedding sidecars that train_wm.sh consumes
#   zero-change.  See docs/superpowers/specs/2026-06-16-rlinf-vectorized-rollout-migration.md.
#
# Usage:
#   # 2 GPUs x 8 envs/GPU, all libero_goal tasks, 300 episodes each:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_collect_rollouts.sh \
#       task_suite_name=libero_goal task_ids=all episodes_per_task=300 \
#       episode_horizon=300 envs_per_gpu=8 \
#       out_dir=data/datasets/collected/libero_goal
#
#   # 1-GPU smoke (single in-process env per rank when envs_per_gpu=1, or K subprocs):
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_rollouts.sh \
#       task_suite_name=libero_goal task_ids=0 episodes_per_task=2 \
#       episode_horizon=20 envs_per_gpu=2 out_dir=/tmp/collect_smoke
#
# Knobs (forwarded as key=value): task_suite_name, task_ids (csv ints or "all"),
#   episodes_per_task, episode_horizon, envs_per_gpu (K), out_dir, unnorm_key, model_path.
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"  # cwd LIBERO paths, not stale ~/.libero
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
cd "${DVLA_ROOT}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_GPUS="${NUM_GPUS:-2}"
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m dreamervla.runners.collect_parallel_rollouts num_gpus="${NUM_GPUS}" "$@"
