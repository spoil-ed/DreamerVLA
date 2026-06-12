#!/usr/bin/env bash
# One-command asset downloader for DreamerVLA.
#
# Each concrete download lives under scripts/download/ so individual model
# families and datasets can be resumed or run by hand.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
DOWNLOAD_SCRIPT_DIR="${DOWNLOAD_SCRIPT_DIR:-${SCRIPT_DIR}/download}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DOWNLOAD_CALVIN="${DOWNLOAD_CALVIN:-0}"
DOWNLOAD_ONLY="${DOWNLOAD_ONLY:-}"
cd "${DVLA_ROOT}"

DOWNLOAD_STEPS=()
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  DOWNLOAD_STEPS+=("10_worldvla.sh" "20_lumina.sh" "30_rynnvla.sh")
fi
if [[ "${DOWNLOAD_LIBERO}" == "1" ]]; then
  DOWNLOAD_STEPS+=("40_libero_dataset.sh")
fi
if [[ "${DOWNLOAD_CALVIN}" == "1" ]]; then
  DOWNLOAD_STEPS+=("50_calvin_dataset.sh")
fi

step_selected() {
  local step="$1"
  local item
  [[ -z "${DOWNLOAD_ONLY}" ]] && return 0
  for item in ${DOWNLOAD_ONLY//,/ }; do
    [[ "${step}" == "${item}" || "${step}" == "${item}"*.sh ]] && return 0
  done
  return 1
}

run_step() {
  local step="$1"
  local script="${DOWNLOAD_SCRIPT_DIR}/${step}"

  if ! step_selected "${step}"; then
    echo "[download_assets] skip ${step} (not selected by DOWNLOAD_ONLY=${DOWNLOAD_ONLY})"
    return
  fi
  if [[ ! -f "${script}" ]]; then
    echo "[download_assets] missing step script: ${script}" >&2
    exit 2
  fi

  echo "[download_assets] start ${step}"
  bash "${script}"
  echo "[download_assets] done ${step}"
}

echo "[download_assets] root=${DVLA_ROOT}"
echo "[download_assets] data_root=${DVLA_DATA_ROOT}"
echo "[download_assets] weights=${DOWNLOAD_WEIGHTS} libero=${DOWNLOAD_LIBERO} calvin=${DOWNLOAD_CALVIN}"

for step in "${DOWNLOAD_STEPS[@]}"; do
  run_step "${step}"
done

echo "[download_assets] complete"
