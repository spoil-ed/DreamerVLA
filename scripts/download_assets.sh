#!/usr/bin/env bash
# One-command asset downloader for DreamerVLA.
#
# Each concrete download lives under scripts/download/ so individual model
# families and datasets can be resumed or run by hand.
#
# How this orchestrator is meant to be extended:
#   1. Add a focused child script under scripts/download/NN_name.sh.
#   2. Put shared knobs in scripts/download/_env.sh only when multiple children
#      need them.
#   3. Append the child script to DOWNLOAD_STEPS below in the exact order it
#      should run.
#   4. Register the script in scripts/README.md and tests/unit_tests.
set -euo pipefail

# Step 0: Resolve repository and data roots.
#
# DVLA_ROOT is the source tree. DVLA_DATA_ROOT is the runtime asset tree. Keep
# them independent: callers may place data on a separate disk or shared volume.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
DOWNLOAD_SCRIPT_DIR="${DOWNLOAD_SCRIPT_DIR:-${SCRIPT_DIR}/download}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_OPENVLA_OFT="${DOWNLOAD_OPENVLA_OFT:-0}"
DOWNLOAD_OPENVLA_ONE_TRAJ="${DOWNLOAD_OPENVLA_ONE_TRAJ:-0}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DOWNLOAD_CALVIN="${DOWNLOAD_CALVIN:-0}"
DOWNLOAD_ONLY="${DOWNLOAD_ONLY:-}"
cd "${DVLA_ROOT}"

DOWNLOAD_STEPS=()
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  DOWNLOAD_STEPS+=("10_rynnvla.sh")
fi
if [[ "${DOWNLOAD_OPENVLA_OFT}" == "1" ]]; then
  DOWNLOAD_STEPS+=("20_openvla_oft.sh")
fi
if [[ "${DOWNLOAD_OPENVLA_ONE_TRAJ}" == "1" ]]; then
  DOWNLOAD_STEPS+=("30_openvla_oft_one_trajectory.sh")
fi
if [[ "${DOWNLOAD_LIBERO}" == "1" ]]; then
  DOWNLOAD_STEPS+=("40_libero_dataset.sh")
fi
if [[ "${DOWNLOAD_CALVIN}" == "1" ]]; then
  DOWNLOAD_STEPS+=("50_calvin_dataset.sh")
fi

# Examples:
#   DOWNLOAD_ONLY=10_rynnvla bash scripts/download_assets.sh
#   DOWNLOAD_OPENVLA_ONE_TRAJ=1 DOWNLOAD_ONLY=30_openvla_oft_one_trajectory bash scripts/download_assets.sh
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
echo "[download_assets] rynnvla=${DOWNLOAD_WEIGHTS} openvla_oft=${DOWNLOAD_OPENVLA_OFT} openvla_one_traj=${DOWNLOAD_OPENVLA_ONE_TRAJ} libero=${DOWNLOAD_LIBERO} calvin=${DOWNLOAD_CALVIN}"
echo "[download_assets] planned_steps=${DOWNLOAD_STEPS[*]:-<none>}"

for step in "${DOWNLOAD_STEPS[@]}"; do
  run_step "${step}"
done

echo "[download_assets] complete"
