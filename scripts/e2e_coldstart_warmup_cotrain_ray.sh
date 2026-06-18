#!/usr/bin/env bash
# E2E launcher: Ray cold-start collection -> offline-warmup online cotrain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

python -m dreamervla.launchers.coldstart_warmup_cotrain --mode ray "$@"
