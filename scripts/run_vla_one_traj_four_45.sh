#!/usr/bin/env bash
# Sequential one-trajectory VLA SFT on GPUs 4,5 for all LIBERO suites.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=(${TASKS:-libero_goal libero_10 libero_object libero_spatial})
TRAJ_OFFSET="${TRAJ_OFFSET:-0}"
TRAJ_PER_TASK="${TRAJ_PER_TASK:-1}"
EXTRA_ARGS=("$@")
mkdir -p data/logs/vla_one_traj_45

for TASK in "${TASKS[@]}"; do
  RUN_TAG="${TASK}_one_traj_o${TRAJ_OFFSET}_gpu45_$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="$(pwd)/data/outputs/vla/pi0_query_one_trajectory/${RUN_TAG}"
  LOG="data/logs/vla_one_traj_45/${RUN_TAG}.log"
  echo "[one_traj_four] start ${TASK} run=${RUN_TAG}"
  TAG="${TASK}" RUN_TAG="${RUN_TAG}" OUT_DIR="${OUT_DIR}" \
    TRAJ_OFFSET="${TRAJ_OFFSET}" TRAJ_PER_TASK="${TRAJ_PER_TASK}" \
    bash scripts/train_vla_one_traj_45.sh \
      training.enable_activation_checkpointing=false \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG}"
  echo "[one_traj_four] done ${TASK}"
done
