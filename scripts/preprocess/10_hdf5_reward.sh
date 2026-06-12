#!/usr/bin/env bash
# Prepare LIBERO HDF5 inputs: write LIBERO config, remove no-op frames, add rewards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-libero_goal}"
OVERWRITE="${OVERWRITE:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export PYTHON
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data"
RAW_LIBERO_DIR="${DVLA_DATA_ROOT}/datasets/libero/${TASK}"
MARKED_DIR="${PROCESSED_DATA_ROOT}/${TASK}_marked_t_256"
HDF5_DIR="${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_256"
REWARD_DIR="${HDF5_DIR}_pi06_remaining_reward"
META_JSON="${PROCESSED_DATA_ROOT}/${TASK}_metainfo.json"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"

mkdir -p "${LIBERO_CONFIG_PATH}" "${PROCESSED_DATA_ROOT}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF

raw_hdf5="$(find "${RAW_LIBERO_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ -z "${raw_hdf5}" ]]; then
  echo "No raw LIBERO HDF5 files found under: ${RAW_LIBERO_DIR}" >&2
  echo "Run: LIBERO_SUITES=${TASK} DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 bash scripts/download_assets.sh" >&2
  exit 2
fi

marked_hdf5="$(find "${MARKED_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ "${OVERWRITE}" == "1" || -z "${marked_hdf5}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${MARKED_DIR}"
  "${PYTHON}" -m dreamer_vla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op \
    --libero_task_suite "${TASK}" \
    --libero_raw_data_dir "${RAW_LIBERO_DIR}" \
    --libero_target_dir "${MARKED_DIR}" \
    --image_resolution 256 \
    --keep-noops
  if [[ -f "${TASK}_metainfo.json" ]]; then
    mv "${TASK}_metainfo.json" "${META_JSON}"
  fi
else
  echo "[10_hdf5_reward] skip mark: ${MARKED_DIR}"
fi

marked_hdf5="$(find "${MARKED_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ -z "${marked_hdf5}" ]]; then
  echo "No marked HDF5 files found under: ${MARKED_DIR}" >&2
  exit 5
fi

filtered_hdf5="$(find "${HDF5_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ "${OVERWRITE}" == "1" || -z "${filtered_hdf5}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${HDF5_DIR}"
  "${PYTHON}" -m dreamer_vla.preprocess.filter_marked_libero_hdf5 \
    --input-dir "${MARKED_DIR}" \
    --output-dir "${HDF5_DIR}" \
    --filter-noops \
    --overwrite
else
  echo "[10_hdf5_reward] skip filter: ${HDF5_DIR}"
fi

filtered_hdf5="$(find "${HDF5_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ -z "${filtered_hdf5}" ]]; then
  echo "No filtered HDF5 files found under: ${HDF5_DIR}" >&2
  exit 5
fi

reward_hdf5="$(find "${REWARD_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)"
if [[ "${OVERWRITE}" == "1" || -z "${reward_hdf5}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${REWARD_DIR}"
  "${PYTHON}" -m dreamer_vla.preprocess.preprocess_remaining_steps_reward \
    --input-dir "${HDF5_DIR}" \
    --output-dir "${REWARD_DIR}" \
    --metainfo-json "${META_JSON}" \
    --overwrite
else
  echo "[10_hdf5_reward] skip reward: ${REWARD_DIR}"
fi
