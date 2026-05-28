#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/data/spoil/workspace/DreamerVLA"
cd "${PROJECT_ROOT}"

WAIT_SESSION="${WAIT_SESSION:-libero_missing45}"
LOG_DIR="${PROJECT_ROOT}/data/logs/vla_nongoal_45"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
  echo "[$(date)] waiting for ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export GPUS="${GPUS:-4,5}"
export PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-4,5}"
export NGPU="${NGPU:-2}"

for task in libero_10 libero_object libero_spatial; do
  ts="$(date +%Y%m%d_%H%M%S)"
  export RUN_TAG="${task}_gpu45_${ts}"
  log="${LOG_DIR}/${RUN_TAG}.log"
  echo "[$(date)] train ${task}; log=${log}"
  TAG="${task}" bash scripts/train_vla_nongoal_45.sh 2>&1 | tee "${log}"
done
