#!/usr/bin/env bash
# Sequential pipeline for the three non-goal LIBERO suites that mirrors the
# pi0-query VLA recipe already used for libero_goal.
#
# For each of libero_10 / libero_object / libero_spatial:
#   1. Generate the his_1 / horizon_1 pretokenized data (Stages 3 + 4abc of
#      prepare_libero_suite_pipeline.sh). PretokenizeActionChunkDataset re-chunks
#      to the per-suite action horizon at load time, so we only need atomic data.
#   2. Run VLA SFT with action_head_type=pi0_query, mirroring the goal recipe
#      (config: pretokenize_vla_libero_<suite>_pi0_query).
#
# Stage 1 (no-op filter) and Stage 2 (pi0.6 remaining reward) are skipped via
# the existence check in prepare_libero_suite_pipeline.sh because their outputs
# already exist for all suites. Stage 5 (action-hidden sidecar) is skipped via
# SKIP_ACTION_HIDDEN=1 — that is only needed for downstream WM, not VLA SFT.
#
# All work is pinned to GPUs 2 and 3.
#
# Usage:
#   bash scripts/train_pi0_query_vla_nongoal.sh
#   SUITES="libero_object" bash scripts/train_pi0_query_vla_nongoal.sh
#   bash scripts/train_pi0_query_vla_libero_10.sh
#   bash scripts/train_pi0_query_vla_libero_object.sh
#   bash scripts/train_pi0_query_vla_libero_spatial.sh
#   SKIP_PREP=1 bash scripts/train_pi0_query_vla_nongoal.sh   # only run SFT
#   SKIP_SFT=1  bash scripts/train_pi0_query_vla_nongoal.sh   # only prep data
#
# Recommended launch (multi-day run, detached):
#   tmux new -d -s pi0q_nongoal \
#     "bash scripts/train_pi0_query_vla_nongoal.sh 2>&1 \
#      | tee data/logs/pi0_query_vla_nongoal/_master_$(date +%Y%m%d_%H%M%S).log"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/data/logs/pi0_query_vla_nongoal}"
mkdir -p "${LOG_DIR}"

# Some downstream scripts hard-code `python`; force the dreamervla conda env.
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHON="${PYTHON:-${CONDA_ENV_BIN}/python}"
hash -r

# Default to GPUs 4,5 for both prep (Chameleon VQGAN encoding) and SFT.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-4,5}"

SUITES_DEFAULT="libero_10 libero_object libero_spatial"
SUITES="${SUITES:-${SUITES_DEFAULT}}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
MASTER_PORT="${MASTER_PORT:-29547}"

# action_horizon=1 in the pretokenized data so PretokenizeActionChunkDataset can
# re-chunk to any per-suite horizon at load time (matches goal's _1_256 atoms).
DATA_HORIZON=1

run_suite_prep () {
  local suite="$1"
  local task_name="${suite#libero_}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log="${LOG_DIR}/${suite}_prep_${stamp}.log"
  echo "[$(date)] === DATA PREP ${suite} -> ${log} ==="
  LIBERO_TASK_SUITE="${suite}" \
  TASK_NAME="${task_name}" \
  LIBERO_TASK_NAME="${task_name}" \
  ACTION_HORIZON="${DATA_HORIZON}" \
  TIME_HORIZON="${DATA_HORIZON}" \
  SKIP_IMAGE_TREE="${SKIP_IMAGE_TREE:-0}" \
  SKIP_PRETOKENIZE=0 \
  SKIP_ACTION_HIDDEN=1 \
  PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES}" \
    bash scripts/prepare_libero_suite_pipeline.sh \
    2>&1 | tee "${log}"
  echo "[$(date)] === DATA PREP done ${suite} ==="
}

run_suite_sft () {
  local suite="$1"
  local horizon="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log="${LOG_DIR}/${suite}_pi0query_sft_${stamp}.log"
  echo "[$(date)] === SFT pi0_query ${suite} (action_horizon=${horizon}) -> ${log} ==="
  VLA_INIT_TAG="${suite}" \
  CONFIG_NAME="pretokenize_vla_${suite}_pi0_query" \
  ACTION_HEAD_TYPE=pi0_query \
  ACTION_HORIZON="${horizon}" \
  NUM_GPUS="${NUM_GPUS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${MASTER_PORT}" \
    bash scripts/pretokenize_train_vla.sh \
    training.gradient_accumulate_every=2 \
    2>&1 | tee "${log}"
  echo "[$(date)] === SFT done ${suite} ==="
}

echo "Pipeline starting at $(date)"
echo "  SUITES   = ${SUITES}"
echo "  GPUs     = ${CUDA_VISIBLE_DEVICES}  (NUM_GPUS=${NUM_GPUS})"
echo "  SKIP_PREP= ${SKIP_PREP}   SKIP_SFT=${SKIP_SFT}"
echo "  LOG_DIR  = ${LOG_DIR}"

for suite in ${SUITES}; do
  case "${suite}" in
    libero_10|libero_spatial) horizon=10 ;;
    libero_object)            horizon=5  ;;
    libero_goal)              horizon=5  ;;
    *) echo "Unknown suite: ${suite}" >&2; exit 2 ;;
  esac
  if [[ "${SKIP_PREP}" != "1" ]]; then
    run_suite_prep "${suite}"
  else
    echo "[$(date)] === PREP skipped for ${suite} ==="
  fi
  if [[ "${SKIP_SFT}" != "1" ]]; then
    run_suite_sft "${suite}" "${horizon}"
  else
    echo "[$(date)] === SFT skipped for ${suite} ==="
  fi
done

echo "[$(date)] ALL DONE."
