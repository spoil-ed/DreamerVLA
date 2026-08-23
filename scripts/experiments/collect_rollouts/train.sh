#!/usr/bin/env bash
# Collect the real LIBERO trajectories used by mainline training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.train \
  --config collect_rollouts \
  "$@"
