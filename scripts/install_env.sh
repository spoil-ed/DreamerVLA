#!/usr/bin/env bash
# Single-machine DreamerVLA environment installer.
#
# This file is intentionally only an orchestrator. Each install phase lives in
# scripts/install/*.sh so failed installs can be resumed without replaying every
# previous phase.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
INSTALL_SCRIPT_DIR="${INSTALL_SCRIPT_DIR:-${SCRIPT_DIR}/install}"
INSTALL_STATE_DIR="${INSTALL_STATE_DIR:-${DVLA_DATA_ROOT}/install_state}"
INSTALL_FORCE="${INSTALL_FORCE:-0}"
INSTALL_ONLY="${INSTALL_ONLY:-}"

INSTALL_STEPS=(
  "00_apt_tools.sh"
  "10_conda_env.sh"
  "20_python_deps.sh"
  "30_third_party.sh"
  "40_verify.sh"
)

step_selected() {
  local step="$1"
  local item
  [[ -z "${INSTALL_ONLY}" ]] && return 0
  for item in ${INSTALL_ONLY//,/ }; do
    [[ "${step}" == "${item}" || "${step}" == "${item}"*.sh ]] && return 0
  done
  return 1
}

run_step() {
  local step="$1"
  local script="${INSTALL_SCRIPT_DIR}/${step}"
  local marker="${INSTALL_STATE_DIR}/${step%.sh}.done"

  if ! step_selected "${step}"; then
    echo "[install_env] skip ${step} (not selected by INSTALL_ONLY=${INSTALL_ONLY})"
    return
  fi
  if [[ -f "${marker}" && "${INSTALL_FORCE}" != "1" ]]; then
    echo "[install_env] skip ${step} (already done: ${marker})"
    return
  fi
  if [[ ! -f "${script}" ]]; then
    echo "[install_env] missing step script: ${script}" >&2
    exit 2
  fi

  echo "[install_env] start ${step}"
  bash "${script}"
  mkdir -p "${INSTALL_STATE_DIR}"
  touch "${marker}"
  echo "[install_env] done ${step}"
}

echo "[install_env] root=${DVLA_ROOT}"
echo "[install_env] data_root=${DVLA_DATA_ROOT}"
echo "[install_env] state_dir=${INSTALL_STATE_DIR}"
echo "[install_env] force=${INSTALL_FORCE} only=${INSTALL_ONLY:-<all>}"

mkdir -p "${INSTALL_STATE_DIR}"
for step in "${INSTALL_STEPS[@]}"; do
  run_step "${step}"
done

echo "[install_env] complete"
