#!/usr/bin/env bash
# Shared environment for DreamerVLA LIBERO preprocessing steps.
set -euo pipefail

PREPROCESS_STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${PREPROCESS_STEP_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"

case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

TASK="${TASK:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
HIS="${HIS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"
FILTER_NOOPS="${FILTER_NOOPS:-1}"
OVERWRITE="${OVERWRITE:-0}"
FORCE="${FORCE:-${OVERWRITE}}"
GPUS="${GPUS:-0}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-16}"
OVERWRITE_TOKENS="${OVERWRITE_TOKENS:-0}"
RUN_MARKED="${RUN_MARKED:-1}"
RUN_REWARD="${RUN_REWARD:-1}"
RUN_PRETOKENIZE="${RUN_PRETOKENIZE:-1}"
RUN_ACTION_HIDDEN="${RUN_ACTION_HIDDEN:-1}"
RUN_INPUT_TOKEN_HIDDEN="${RUN_INPUT_TOKEN_HIDDEN:-0}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"

case "${TASK}" in
  libero_goal|libero_object) DEFAULT_TIME_HORIZON=5 ;;
  libero_spatial|libero_10) DEFAULT_TIME_HORIZON=10 ;;
  *) DEFAULT_TIME_HORIZON="${ACTION_HORIZON}" ;;
esac
TIME_HORIZON="${TIME_HORIZON:-${DEFAULT_TIME_HORIZON}}"

PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${DVLA_DATA_ROOT}/processed_data}"
RAW_LIBERO_DIR="${RAW_LIBERO_DIR:-${DVLA_DATA_ROOT}/datasets/libero/${TASK}}"
MARKED_DIR="${MARKED_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_marked_t_${IMAGE_RESOLUTION}}"
if [[ "${FILTER_NOOPS}" == "1" ]]; then
  HDF5_DIR="${HDF5_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_${IMAGE_RESOLUTION}}"
else
  HDF5_DIR="${HDF5_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_with_noops_t_${IMAGE_RESOLUTION}}"
fi
REWARD_DIR="${REWARD_DIR:-${HDF5_DIR}_pi06_remaining_reward}"
HIDDEN_DIR="${HIDDEN_DIR:-${HDF5_DIR}_pi0_legacy_action_hidden_vla_policy_h2}"
IMG_STATE_DIR="${IMG_STATE_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_image_state_action_t_${IMAGE_RESOLUTION}}"
META_JSON="${META_JSON:-${PROCESSED_DATA_ROOT}/${TASK}_metainfo.json}"
VLA_CKPT="${VLA_CKPT:-${DVLA_DATA_ROOT}/checkpoints/VLA_model_256/${TASK}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DVLA_DATA_ROOT}/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768}"
TEXT_TOKENIZER_PATH="${TEXT_TOKENIZER_PATH:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/text_tokenizer.json}"
CHAMELEON_VQGAN_CONFIG="${CHAMELEON_VQGAN_CONFIG:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.yaml}"
CHAMELEON_VQGAN_CKPT="${CHAMELEON_VQGAN_CKPT:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.ckpt}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-1}}"
VALIDATE_ACTION_HIDDEN="${VALIDATE_ACTION_HIDDEN:-0}"

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
CONFIG_DIR="${DVLA_DATA_ROOT}/configs/${TASK}"
LOG_DIR="${DVLA_DATA_ROOT}/logs/libero_data_prep"
TASK_NAME="${TASK#libero_}"
SUFFIX="his_${HIS}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
MANIFEST="${CONCATE_DIR}/${TASK}_${SUFFIX}.json"
VAL_IND_REC="${TOKENS_DIR}/${TASK}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"
VAL_OOD_REC="${TOKENS_DIR}/${TASK}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"

cd "${DVLA_ROOT}"
mkdir -p "${PROCESSED_DATA_ROOT}" "${CONVS_DIR}" "${TOKENS_DIR}" "${CONCATE_DIR}" "${CONFIG_DIR}" "${LOG_DIR}"

preprocess_log() {
  printf '[preprocess:%s] %s\n' "$(basename "$0")" "$*"
}

write_libero_config() {
  mkdir -p "${LIBERO_CONFIG_PATH}"
  if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
    cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
    preprocess_log "wrote LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml"
  else
    preprocess_log "skipped LIBERO config write"
  fi
}

has_hdf5_files() {
  local dir="$1"
  local found=""
  [[ -d "${dir}" ]] || return 1
  found="$(find "${dir}" -maxdepth 1 -type f -name '*.hdf5' -print -quit)"
  [[ -n "${found}" ]]
}

hdf5_count() {
  local dir="$1"
  [[ -d "${dir}" ]] || { echo 0; return; }
  find "${dir}" -maxdepth 1 -type f -name '*.hdf5' | wc -l
}

child_dir_count() {
  local dir="$1"
  [[ -d "${dir}" ]] || { echo 0; return; }
  find "${dir}" -mindepth 1 -maxdepth 1 -type d | wc -l
}

has_regular_files() {
  local dir="$1"
  local found=""
  [[ -d "${dir}" ]] || return 1
  found="$(find "${dir}" -maxdepth 1 -type f -print -quit)"
  [[ -n "${found}" ]]
}

require_hdf5_files() {
  local dir="$1"
  local message="$2"
  local code="${3:-5}"
  if ! has_hdf5_files "${dir}"; then
    echo "${message}: ${dir}" >&2
    exit "${code}"
  fi
}

require_file() {
  local path="$1"
  local message="$2"
  local code="${3:-4}"
  if [[ ! -f "${path}" ]]; then
    echo "${message}: ${path}" >&2
    exit "${code}"
  fi
}

json_len() {
  [[ -f "$1" ]] || { echo 0; return; }
  "${PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$1" 2>/dev/null || echo 0
}

pkl_count() {
  [[ -d "$1/files" ]] || { echo 0; return; }
  find "$1/files" -maxdepth 1 -type f -name '*.pkl' | wc -l
}

normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}
