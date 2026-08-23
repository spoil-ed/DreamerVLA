#!/usr/bin/env bash
# Train the configured world model on one LIBERO trajectory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.diagnostics.wm_single_trajectory_overfit --run "$@"
