#!/usr/bin/env bash
# Train the staged full VLA + world-model + classifier cotrain route.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.frozen_model_cotrain_ray \
  experiment=dreamervla_wmcls_cotrain_ray \
  "$@"
