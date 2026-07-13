#!/usr/bin/env bash
# Train the official-data Chunk world model through Hydra.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.train \
  --config-name world_model_training \
  "$@"
