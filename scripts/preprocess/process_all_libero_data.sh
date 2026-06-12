#!/usr/bin/env bash
# Compatibility wrapper for the lower LIBERO image/conversation/token/config path.
#
# New code should prefer scripts/preprocess/prepare_libero_data.sh or the
# numbered step scripts directly. This wrapper is kept for existing workflows
# that already prepared final HDF5/reward files and want to process one or more
# suites through the image -> conv -> token -> config stages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PREPROCESS_SCRIPT_DIR="${PREPROCESS_SCRIPT_DIR:-${SCRIPT_DIR}}"
PROCESS_ALL_ONLY="${PROCESS_ALL_ONLY:-${PREPROCESS_ONLY:-}}"
cd "${DVLA_ROOT}"

SUITES="${SUITES:-libero_10 libero_object libero_spatial}"
HIS="${HIS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"

PROCESS_ALL_STEPS=(
  "20_pretokenize_dataset.sh"
)

step_selected() {
  local step="$1"
  local item
  [[ -z "${PROCESS_ALL_ONLY}" ]] && return 0
  for item in ${PROCESS_ALL_ONLY//,/ }; do
    [[ "${step}" == "${item}" || "${step}" == "${item}"*.sh ]] && return 0
  done
  return 1
}

run_step() {
  local suite="$1"
  local step="$2"
  local script="${PREPROCESS_SCRIPT_DIR}/${step}"

  if ! step_selected "${step}"; then
    echo "[process_all_libero_data] skip ${suite} ${step} (not selected by PROCESS_ALL_ONLY=${PROCESS_ALL_ONLY})"
    return
  fi
  if [[ ! -f "${script}" ]]; then
    echo "[process_all_libero_data] missing step script: ${script}" >&2
    exit 2
  fi

  echo "[process_all_libero_data] start ${suite} ${step}"
  set +e
  TASK="${suite}" \
  HIS="${HIS}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  IMAGE_RESOLUTION="${IMAGE_RESOLUTION}" \
    bash "${script}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[process_all_libero_data] FAIL ${suite} ${step} exit=${rc}"
    return "${rc}"
  fi
  echo "[process_all_libero_data] done ${suite} ${step}"
}

echo "[process_all_libero_data] root=${DVLA_ROOT}"
echo "[process_all_libero_data] data_root=${DVLA_DATA_ROOT}"
echo "[process_all_libero_data] suites=${SUITES}"
echo "[process_all_libero_data] planned_steps=${PROCESS_ALL_STEPS[*]}"
echo "[process_all_libero_data] resume_hint=use PROCESS_ALL_ONLY=<step> or run scripts/preprocess/<step> directly"

OVERALL_RC=0
for suite in ${SUITES}; do
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " SUITE = ${suite}"
  echo "════════════════════════════════════════════════════════════════"
  if ! (
    set -e
    for step in "${PROCESS_ALL_STEPS[@]}"; do
      if ! run_step "${suite}" "${step}"; then
        exit 1
      fi
    done
  ); then
    echo "✗ ${suite} FAILED"
    OVERALL_RC=1
  else
    echo "✓ ${suite} DONE"
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " ALL DONE  (overall_rc=${OVERALL_RC})  $(date)"
echo "════════════════════════════════════════════════════════════════"
for suite in libero_goal libero_10 libero_object libero_spatial; do
  manifest="${DVLA_DATA_ROOT}/processed_data/concate_tokens/${suite}_his_${HIS}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
  if [[ -f "${manifest}" ]]; then
    n=$("${PYTHON:-python}" -c "import json; print(len(json.load(open('${manifest}'))))" 2>/dev/null)
    echo "  ${suite}: ${n} manifest entries"
  else
    echo "  ${suite}: MISSING manifest"
  fi
done
exit "${OVERALL_RC}"
