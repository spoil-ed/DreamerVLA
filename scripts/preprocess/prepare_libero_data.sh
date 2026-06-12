#!/usr/bin/env bash
# One-command LIBERO preprocessing for the DreamerVLA data layout.
set -euo pipefail

# ---- environment -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
cd "${DVLA_ROOT}"

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

TASK="${TASK:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
HIS="${HIS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"
FILTER_NOOPS="${FILTER_NOOPS:-1}"
RUN_MARKED="${RUN_MARKED:-1}"
RUN_REWARD="${RUN_REWARD:-1}"
RUN_PRETOKENIZE="${RUN_PRETOKENIZE:-1}"
RUN_ACTION_HIDDEN="${RUN_ACTION_HIDDEN:-1}"
OVERWRITE="${OVERWRITE:-0}"

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
META_JSON="${META_JSON:-${PROCESSED_DATA_ROOT}/${TASK}_metainfo.json}"
VLA_CKPT="${VLA_CKPT:-${DVLA_DATA_ROOT}/checkpoints/VLA_model_256/${TASK}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DVLA_DATA_ROOT}/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768}"
TEXT_TOKENIZER_PATH="${TEXT_TOKENIZER_PATH:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/text_tokenizer.json}"
CHAMELEON_VQGAN_CONFIG="${CHAMELEON_VQGAN_CONFIG:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.yaml}"
CHAMELEON_VQGAN_CKPT="${CHAMELEON_VQGAN_CKPT:-${DVLA_DATA_ROOT}/checkpoints/chameleon/tokenizer/vqgan.ckpt}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-1}}"

mkdir -p "${PROCESSED_DATA_ROOT}" "${DVLA_DATA_ROOT}/logs/libero_data_prep"

has_hdf5_files() {
  local dir="$1"
  local found=""
  [[ -d "${dir}" ]] || return 1
  found="$(find "${dir}" -maxdepth 1 -type f -name '*.hdf5' -print -quit)"
  [[ -n "${found}" ]]
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

echo "[prepare_libero_data] task=${TASK} his=${HIS} len_action=${ACTION_HORIZON} resolution=${IMAGE_RESOLUTION}"
echo "[prepare_libero_data] raw=${RAW_LIBERO_DIR}"
echo "[prepare_libero_data] marked=${MARKED_DIR}"
echo "[prepare_libero_data] hdf5=${HDF5_DIR}"
echo "[prepare_libero_data] reward=${REWARD_DIR}"
echo "[prepare_libero_data] hidden=${HIDDEN_DIR}"

if [[ ! -d "${RAW_LIBERO_DIR}" ]]; then
  echo "Missing raw LIBERO dataset dir: ${RAW_LIBERO_DIR}" >&2
  echo "Run: LIBERO_SUITES=${TASK} DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 bash scripts/download_assets.sh" >&2
  exit 2
fi
if ! has_hdf5_files "${RAW_LIBERO_DIR}"; then
  echo "No raw LIBERO HDF5 files found under: ${RAW_LIBERO_DIR}" >&2
  echo "Run: LIBERO_SUITES=${TASK} DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 bash scripts/download_assets.sh" >&2
  exit 2
fi

if [[ "${RUN_MARKED}" == "1" ]]; then
  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${MARKED_DIR}"; then
    echo "[prepare_libero_data] stage 1: replay and mark no-ops"
    if [[ -d "${MARKED_DIR}" ]]; then
      echo "[prepare_libero_data] removing incomplete or overwritten marked dir: ${MARKED_DIR}"
      rm -rf "${MARKED_DIR}"
    fi
    "${PYTHON}" -m dreamer_vla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op \
      --libero_task_suite "${TASK}" \
      --libero_raw_data_dir "${RAW_LIBERO_DIR}" \
      --libero_target_dir "${MARKED_DIR}" \
      --image_resolution "${IMAGE_RESOLUTION}" \
      --keep-noops
    if [[ -f "${TASK}_metainfo.json" ]]; then
      mv "${TASK}_metainfo.json" "${META_JSON}"
    fi
  else
    echo "[prepare_libero_data] stage 1 skipped: ${MARKED_DIR}"
  fi
  require_hdf5_files "${MARKED_DIR}" "[prepare_libero_data] stage 1 did not create marked HDF5 files"

  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${HDF5_DIR}"; then
    echo "[prepare_libero_data] stage 2: create final HDF5 view"
    if [[ -d "${HDF5_DIR}" ]]; then
      echo "[prepare_libero_data] removing incomplete or overwritten HDF5 dir: ${HDF5_DIR}"
      rm -rf "${HDF5_DIR}"
    fi
    filter_arg=()
    [[ "${FILTER_NOOPS}" == "1" ]] && filter_arg=(--filter-noops)
    "${PYTHON}" -m dreamer_vla.preprocess.filter_marked_libero_hdf5 \
      --input-dir "${MARKED_DIR}" \
      --output-dir "${HDF5_DIR}" \
      --overwrite \
      "${filter_arg[@]}"
  else
    echo "[prepare_libero_data] stage 2 skipped: ${HDF5_DIR}"
  fi
  require_hdf5_files "${HDF5_DIR}" "[prepare_libero_data] stage 2 did not create final HDF5 files"
fi

if [[ "${RUN_REWARD}" == "1" ]]; then
  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${REWARD_DIR}"; then
    echo "[prepare_libero_data] stage 3: remaining-steps reward"
    if [[ -d "${REWARD_DIR}" ]]; then
      echo "[prepare_libero_data] removing incomplete or overwritten reward dir: ${REWARD_DIR}"
      rm -rf "${REWARD_DIR}"
    fi
    reward_args=(--input-dir "${HDF5_DIR}" --output-dir "${REWARD_DIR}" --overwrite)
    [[ -f "${META_JSON}" ]] && reward_args+=(--metainfo-json "${META_JSON}")
    "${PYTHON}" -m dreamer_vla.preprocess.preprocess_remaining_steps_reward "${reward_args[@]}"
  else
    echo "[prepare_libero_data] stage 3 skipped: ${REWARD_DIR}"
  fi
  require_hdf5_files "${REWARD_DIR}" "[prepare_libero_data] stage 3 did not create reward HDF5 files"
fi

if [[ "${RUN_PRETOKENIZE}" == "1" ]]; then
  if [[ "${FILTER_NOOPS}" != "1" ]]; then
    echo "Pretokenize configs currently target *_no_noops_t_* paths; set FILTER_NOOPS=1 or RUN_PRETOKENIZE=0." >&2
    exit 3
  fi
  if ! has_regular_files "${TOKENIZER_PATH}"; then
    echo "Missing Lumina tokenizer/backbone files for pretokenization: ${TOKENIZER_PATH}" >&2
    echo "Run: DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
    exit 4
  fi
  echo "[prepare_libero_data] stage 4: image tree, convs, tokens, configs"
  SUITES="${TASK}" \
  HIS="${HIS}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
  FORCE="${OVERWRITE}" \
    bash "${DVLA_ROOT}/scripts/preprocess/process_all_libero_data.sh"
fi

if [[ "${RUN_ACTION_HIDDEN}" == "1" ]]; then
  if ! has_regular_files "${VLA_CKPT}"; then
    echo "Missing VLA checkpoint files for action-hidden sidecar: ${VLA_CKPT}" >&2
    echo "Run: LIBERO_SUITES=${TASK} DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
    exit 4
  fi
  if ! has_regular_files "${TOKENIZER_PATH}"; then
    echo "Missing Lumina tokenizer/backbone files for preprocessing: ${TOKENIZER_PATH}" >&2
    echo "Run: DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
    exit 4
  fi
  require_file "${TEXT_TOKENIZER_PATH}" "Missing Chameleon text tokenizer for preprocessing" 4
  require_file "${CHAMELEON_VQGAN_CONFIG}" "Missing Chameleon VQGAN config for preprocessing" 4
  require_file "${CHAMELEON_VQGAN_CKPT}" "Missing Chameleon VQGAN checkpoint for preprocessing" 4
  if [[ ! -d "${HIDDEN_DIR}" || "${OVERWRITE}" == "1" ]]; then
    echo "[prepare_libero_data] stage 5: legacy action-hidden sidecar"
    [[ "${OVERWRITE}" == "1" ]] && rm -rf "${HIDDEN_DIR}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node="${ACTION_HIDDEN_GPUS}" \
      --module dreamer_vla.preprocess.preprocess_rynn_pixel_hidden \
      --hdf5-dir "${REWARD_DIR}" \
      --out-dir "${HIDDEN_DIR}" \
      --model-path "${VLA_CKPT}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --text-tokenizer-path "${TEXT_TOKENIZER_PATH}" \
      --chameleon-vqgan-config "${CHAMELEON_VQGAN_CONFIG}" \
      --chameleon-vqgan-ckpt "${CHAMELEON_VQGAN_CKPT}" \
      --action-head-type legacy \
      --obs-hidden-source action_query \
      --history 2 \
      --include-state \
      --rotate-images-180 \
      --save-action-hidden \
      --action-dim 7 \
      --time-horizon "${TIME_HORIZON}" \
      --overwrite
  else
    echo "[prepare_libero_data] stage 5 skipped: ${HIDDEN_DIR}"
  fi
fi

echo "[prepare_libero_data] complete"
