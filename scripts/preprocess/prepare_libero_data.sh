#!/usr/bin/env bash
# One-command LIBERO preprocessing orchestrator.
#
# Each concrete phase lives in scripts/preprocess/NN_name.sh so failed stages
# can be rerun directly, mirroring scripts/install/ and scripts/download/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PREPROCESS_SCRIPT_DIR="${PREPROCESS_SCRIPT_DIR:-${SCRIPT_DIR}}"
PREPROCESS_ONLY="${PREPROCESS_ONLY:-}"

RUN_MARKED="${RUN_MARKED:-1}"
RUN_REWARD="${RUN_REWARD:-1}"
RUN_PRETOKENIZE="${RUN_PRETOKENIZE:-1}"
RUN_ACTION_HIDDEN="${RUN_ACTION_HIDDEN:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
export RUN_MARKED RUN_REWARD RUN_PRETOKENIZE RUN_ACTION_HIDDEN RUN_VALIDATE
export VALIDATE_ACTION_HIDDEN="${VALIDATE_ACTION_HIDDEN:-${RUN_ACTION_HIDDEN}}"

source "${SCRIPT_DIR}/_env.sh"

PREPROCESS_STEPS=()
if [[ "${RUN_MARKED}" == "1" || "${RUN_REWARD}" == "1" ]]; then
  PREPROCESS_STEPS+=("10_hdf5_reward.sh")
fi
if [[ "${RUN_PRETOKENIZE}" == "1" ]]; then
  PREPROCESS_STEPS+=("20_pretokenize_dataset.sh")
fi
if [[ "${RUN_ACTION_HIDDEN}" == "1" ]]; then
  PREPROCESS_STEPS+=("30_action_hidden.sh")
fi
if [[ "${RUN_VALIDATE}" == "1" && "${RUN_PRETOKENIZE}" == "1" ]]; then
  PREPROCESS_STEPS+=("40_validate.sh")
fi

step_selected() {
  local step="$1"
  local item
  [[ -z "${PREPROCESS_ONLY}" ]] && return 0
  for item in ${PREPROCESS_ONLY//,/ }; do
    [[ "${step}" == "${item}" || "${step}" == "${item}"*.sh ]] && return 0
  done
  return 1
}

run_step() {
  local step="$1"
  local script="${PREPROCESS_SCRIPT_DIR}/${step}"

  if ! step_selected "${step}"; then
    echo "[prepare_libero_data] skip ${step} (not selected by PREPROCESS_ONLY=${PREPROCESS_ONLY})"
    return
  fi
  if [[ ! -f "${script}" ]]; then
    echo "[prepare_libero_data] missing step script: ${script}" >&2
    exit 2
  fi

  echo "[prepare_libero_data] start ${step}"
  bash "${script}"
  echo "[prepare_libero_data] done ${step}"
}

echo "[prepare_libero_data] root=${DVLA_ROOT}"
echo "[prepare_libero_data] data_root=${DVLA_DATA_ROOT}"
echo "[prepare_libero_data] task=${TASK} his=${HIS} len_action=${ACTION_HORIZON} resolution=${IMAGE_RESOLUTION}"
echo "[prepare_libero_data] raw=${RAW_LIBERO_DIR}"
echo "[prepare_libero_data] hdf5=${HDF5_DIR}"
echo "[prepare_libero_data] reward=${REWARD_DIR}"
echo "[prepare_libero_data] hidden=${HIDDEN_DIR}"
echo "[prepare_libero_data] planned_steps=${PREPROCESS_STEPS[*]}"
echo "[prepare_libero_data] resume_hint=use PREPROCESS_ONLY=<step> or run scripts/preprocess/<step> directly"

for step in "${PREPROCESS_STEPS[@]}"; do
  run_step "${step}"
done

echo "[prepare_libero_data] complete"
