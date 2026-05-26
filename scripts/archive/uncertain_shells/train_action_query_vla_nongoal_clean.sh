#!/usr/bin/env bash
# Clean non-goal VLA SFT launcher.
#
# This wrapper keeps every suite pinned to its own RynnVLA-002 checkpoint and
# avoids inheriting the libero_goal defaults from env_libero_goal.sh.
#
# Usage:
#   bash scripts/train_action_query_vla_nongoal_clean.sh
#   SUITES="libero_10" bash scripts/train_action_query_vla_nongoal_clean.sh
#   SKIP_PREP=1 bash scripts/train_action_query_vla_nongoal_clean.sh
#   SKIP_SFT=1 bash scripts/train_action_query_vla_nongoal_clean.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/data/logs/action_query_vla_nongoal}"
mkdir -p "${LOG_DIR}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHON="${PYTHON:-${CONDA_ENV_BIN}/python}"
hash -r

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export PREPROCESS_GPU_DEVICES="${PREPROCESS_GPU_DEVICES:-4,5}"

SUITES="${SUITES:-libero_10 libero_object libero_spatial}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
MASTER_PORT="${MASTER_PORT:-29547}"
DATA_HORIZON=1

suite_horizon() {
  case "$1" in
    libero_goal|libero_object) echo 5 ;;
    libero_10|libero_spatial)  echo 10 ;;
    *) echo "Unknown suite: $1" >&2; return 2 ;;
  esac
}

suite_task_name() {
  printf '%s\n' "${1#libero_}"
}

suite_ckpt() {
  printf '%s\n' "${PROJECT_ROOT}/data/ckpts/VLA_model_256/$1"
}

require_suite_ckpt() {
  local suite="$1"
  local horizon="$2"
  local ckpt
  ckpt="$(suite_ckpt "${suite}")"
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "ERROR: missing suite checkpoint config: ${ckpt}/config.json" >&2
    exit 3
  fi
  local ckpt_horizon
  ckpt_horizon="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["time_horizon"])' "${ckpt}/config.json")"
  if [[ "${ckpt_horizon}" != "${horizon}" ]]; then
    echo "ERROR: ${suite} horizon mismatch: launcher=${horizon}, ckpt=${ckpt_horizon} (${ckpt}/config.json)" >&2
    exit 4
  fi
}

run_suite_prep() {
  local suite="$1"
  local task_name
  task_name="$(suite_task_name "${suite}")"
  local stamp log
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_DIR}/${suite}_prep_${stamp}.log"
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

run_suite_sft() {
  local suite="$1"
  local horizon="$2"
  local ckpt
  ckpt="$(suite_ckpt "${suite}")"
  local stamp log
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_DIR}/${suite}_action_query_sft_${stamp}.log"
  echo "[$(date)] === SFT action_query ${suite} (action_horizon=${horizon}, ckpt=${ckpt}) -> ${log} ==="

  VLA_INIT_TAG="${suite}" \
  VLA_INIT_CKPT="${ckpt}" \
  MODEL_PATH="${ckpt}" \
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

echo "Clean action-query VLA pipeline starting at $(date)"
echo "  SUITES   = ${SUITES}"
echo "  GPUs     = ${CUDA_VISIBLE_DEVICES}  (NUM_GPUS=${NUM_GPUS})"
echo "  SKIP_PREP= ${SKIP_PREP}   SKIP_SFT=${SKIP_SFT}"
echo "  LOG_DIR  = ${LOG_DIR}"

for suite in ${SUITES}; do
  horizon="$(suite_horizon "${suite}")"
  ckpt="$(suite_ckpt "${suite}")"
  require_suite_ckpt "${suite}" "${horizon}"
  echo "[$(date)] suite=${suite} horizon=${horizon} ckpt=${ckpt}"

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
