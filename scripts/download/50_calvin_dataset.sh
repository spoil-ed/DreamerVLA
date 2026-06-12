#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
mkdir -p "${CALVIN_DIR}"

for task in $(normalize_list "${CALVIN_TASKS}"); do
  [[ -n "${task}" ]] || continue
  zip_path="${CALVIN_DIR}/${task}.zip"
  if [[ ! -f "${zip_path}" ]]; then
    download_log "CALVIN ${task} archive -> ${zip_path}"
    curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${zip_path}"
  else
    download_log "CALVIN ${task} archive already exists: ${zip_path}"
  fi
  if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
    download_log "extract CALVIN ${task} -> ${CALVIN_DIR}/${task}"
    "${PYTHON}" -m zipfile -e "${zip_path}" "${CALVIN_DIR}/${task}"
  fi
done
