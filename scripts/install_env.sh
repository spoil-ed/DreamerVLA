#!/usr/bin/env bash
# Hydra entrypoint: DreamerVLA environment install workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

if ! python -c "import hydra, omegaconf" >/dev/null 2>&1; then
  python -m pip install --user hydra-core omegaconf
fi

python -m dreamervla.launchers.workflow --config-name install "$@"
