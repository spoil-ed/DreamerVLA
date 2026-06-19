#!/usr/bin/env bash
# Validate LIBERO preprocessing artifacts without opening token pkl payloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
SUITES="${SUITES:-${LIBERO_SUITES:-${TASK:-libero_goal}}}"
SUITES_LIST="[${SUITES// /,}]"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

python -m dreamervla.preprocess.validate_libero_data_prep \
  data_root="${DVLA_DATA_ROOT}" \
  suites="${SUITES_LIST}" \
  "$@"
