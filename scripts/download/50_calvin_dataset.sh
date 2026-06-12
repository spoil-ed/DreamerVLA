#!/usr/bin/env bash
# Download optional CALVIN archives.
#
# Output:
#   ${DVLA_DATA_ROOT}/datasets/calvin/<task>.zip
#   ${DVLA_DATA_ROOT}/datasets/calvin/<task>/  when EXTRACT_CALVIN=1
#
# Add more CALVIN tasks by passing CALVIN_TASKS="task_A task_B"; keep all CALVIN
# files under datasets/calvin/ so they do not mix with LIBERO suites.
set -euo pipefail

# Step 0: Load shared roots and Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

# Step 1: CALVIN source and target task list.
CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
mkdir -p "${CALVIN_DIR}"

# Step 2: Download archives one task at a time; curl -C resumes partial files.
for task in $(normalize_list "${CALVIN_TASKS}"); do
  [[ -n "${task}" ]] || continue
  zip_path="${CALVIN_DIR}/${task}.zip"
  if [[ ! -f "${zip_path}" ]]; then
    download_log "CALVIN ${task} archive -> ${zip_path}"
    curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${zip_path}"
  else
    download_log "CALVIN ${task} archive already exists: ${zip_path}"
  fi

  # Step 3: Extraction is explicit because archives can be large.
  if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
    download_log "extract CALVIN ${task} -> ${CALVIN_DIR}/${task}"
    "${PYTHON}" -m zipfile -e "${zip_path}" "${CALVIN_DIR}/${task}"
  fi
done
