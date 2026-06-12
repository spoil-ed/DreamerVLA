#!/usr/bin/env bash
# Prepare LIBERO HDF5 inputs: config, no-op marking/filtering, and rewards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

write_libero_config
preprocess_log "task=${TASK} raw=${RAW_LIBERO_DIR} marked=${MARKED_DIR} hdf5=${HDF5_DIR} reward=${REWARD_DIR}"

if [[ "${RUN_MARKED}" == "1" ]]; then
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

  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${MARKED_DIR}"; then
    preprocess_log "stage 1: replay and mark no-ops"
    if [[ -d "${MARKED_DIR}" ]]; then
      preprocess_log "removing incomplete or overwritten marked dir: ${MARKED_DIR}"
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
    preprocess_log "stage 1 skipped: ${MARKED_DIR}"
  fi
  require_hdf5_files "${MARKED_DIR}" "[preprocess:10_hdf5_reward.sh] stage 1 did not create marked HDF5 files"

  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${HDF5_DIR}"; then
    preprocess_log "stage 2: create final HDF5 view"
    if [[ -d "${HDF5_DIR}" ]]; then
      preprocess_log "removing incomplete or overwritten HDF5 dir: ${HDF5_DIR}"
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
    preprocess_log "stage 2 skipped: ${HDF5_DIR}"
  fi
  require_hdf5_files "${HDF5_DIR}" "[preprocess:10_hdf5_reward.sh] stage 2 did not create final HDF5 files"
fi

if [[ "${RUN_REWARD}" == "1" ]]; then
  require_hdf5_files "${HDF5_DIR}" "[preprocess:10_hdf5_reward.sh] missing final HDF5 input" 5
  if [[ "${OVERWRITE}" == "1" ]] || ! has_hdf5_files "${REWARD_DIR}"; then
    preprocess_log "stage 3: remaining-steps reward"
    if [[ -d "${REWARD_DIR}" ]]; then
      preprocess_log "removing incomplete or overwritten reward dir: ${REWARD_DIR}"
      rm -rf "${REWARD_DIR}"
    fi
    reward_args=(--input-dir "${HDF5_DIR}" --output-dir "${REWARD_DIR}" --overwrite)
    [[ -f "${META_JSON}" ]] && reward_args+=(--metainfo-json "${META_JSON}")
    "${PYTHON}" -m dreamer_vla.preprocess.preprocess_remaining_steps_reward "${reward_args[@]}"
  else
    preprocess_log "stage 3 skipped: ${REWARD_DIR}"
  fi
  require_hdf5_files "${REWARD_DIR}" "[preprocess:10_hdf5_reward.sh] stage 3 did not create reward HDF5 files"
fi
