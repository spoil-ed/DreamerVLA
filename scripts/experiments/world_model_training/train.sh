#!/usr/bin/env bash
# Train a Hydra-selected world model over official LIBERO data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.train \
  --config dreamer-wm \
  "$@"
