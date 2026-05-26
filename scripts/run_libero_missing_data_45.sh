#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/mnt/data/spoil/workspace/DreamerVLA"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/data/logs/libero_data_prep"
mkdir -p "${LOG_DIR}"

export GPUS="${GPUS:-4,5}"
export PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-${GPUS}}"
export PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-16}"

run_stage() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date)] === ${name} ==="
  echo "log=${log}"
  env "$@" bash scripts/process_all_libero_data.sh 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "[$(date)] ${name} rc=${rc}"
  return "${rc}"
}

run_stage fix_spatial_his1_len1 \
  SUITES=libero_spatial \
  ACTION_HORIZON=1 \
  HIS=1 \
  GPUS="${GPUS}" \
  PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES}" \
  PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS}"

if [[ "${RUN_LEN10:-0}" == "1" ]]; then
  run_stage build_libero10_goal_his1_len10 \
    SUITES="libero_10 libero_goal" \
    ACTION_HORIZON=10 \
    HIS=1 \
    GPUS="${GPUS}" \
    PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES}" \
    PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS}"
fi
