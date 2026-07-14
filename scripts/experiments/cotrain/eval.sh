#!/usr/bin/env bash
# Evaluate one explicit manual-cotrain policy checkpoint through Hydra.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.train \
  --config eval_cotrain \
  "$@"
