#!/usr/bin/env bash
# Validate LIBERO preprocessing artifacts without opening token pkl payloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"
cd "${DVLA_ROOT}"

suite_args=()
if [[ "$#" -eq 0 ]]; then
  if [[ -n "${TASK:-}" ]]; then
    suite_args=(--suites "${TASK}")
  elif [[ -n "${LIBERO_SUITES:-}" ]]; then
    read -r -a suite_values <<< "${LIBERO_SUITES}"
    suite_args=(--suites "${suite_values[@]}")
  elif [[ -n "${SUITES:-}" ]]; then
    read -r -a suite_values <<< "${SUITES}"
    suite_args=(--suites "${suite_values[@]}")
  fi
fi

exec "${PYTHON}" -m dreamer_vla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  "${suite_args[@]}" \
  "$@"
