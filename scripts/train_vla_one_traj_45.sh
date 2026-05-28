#!/usr/bin/env bash
# GPUs 4,5 one-trajectory VLA SFT. Switch task with argv[1] or TAG/TASK.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-${TAG:-libero_spatial}}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  TASK="$1"
  shift
fi

case "${TASK}" in libero_goal|libero_10|libero_object|libero_spatial) ;; *) echo "ERROR: TASK must be one of: libero_goal, libero_10, libero_object, libero_spatial" >&2; exit 2 ;; esac

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHON="${PYTHON:-${CONDA_ENV_BIN}/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" GPUS="${GPUS:-4,5}" PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-4,5}"
export NGPU="${NGPU:-2}"
export NUM_GPUS="${NUM_GPUS:-${NGPU}}"
export MASTER_PORT="${MASTER_PORT:-29545}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

export CONFIG="${CONFIG:-vla_sft_one_trajectory}"
CKPT="${CKPT:-$(pwd)/data/ckpts/VLA_model_256/${TASK}}"
TRAJ_OFFSET="${TRAJ_OFFSET:-0}"
TRAJ_PER_TASK="${TRAJ_PER_TASK:-1}"
RUN_TAG="${RUN_TAG:-${TASK}_one_traj_o${TRAJ_OFFSET}_gpu45_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-$(pwd)/data/outputs/vla/pi0_query_one_trajectory/${RUN_TAG}}"

[[ -f "${CKPT}/config.json" ]] || { echo "ERROR: missing pretrained checkpoint: ${CKPT}/config.json" >&2; exit 3; }
HORIZON="$("${CONDA_ENV_BIN}/python" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("time_horizon") or c.get("action_horizon") or "")' "${CKPT}/config.json")"
[[ -n "${HORIZON}" ]] || { echo "ERROR: missing time_horizon/action_horizon in ${CKPT}/config.json" >&2; exit 4; }

EXTRA=()
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA+=("dataloader.batch_size=${BATCH_SIZE}")

exec bash scripts/train_vla.sh \
  "task=${TASK}" \
  "init.vla_ckpt_path=${CKPT}" \
  "encoder.model_path=${CKPT}" \
  "task.action_horizon=${HORIZON}" \
  "task.time_horizon=${HORIZON}" \
  "encoder.time_horizon=${HORIZON}" \
  "dataset.trajectory_offset=${TRAJ_OFFSET}" \
  "dataset.trajectories_per_task=${TRAJ_PER_TASK}" \
  "${EXTRA[@]}" \
  "$@"
