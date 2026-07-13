#!/usr/bin/env bash
# Validate inputs and summarize the selected single-trajectory overfit run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.diagnostics.wm_single_trajectory_overfit "$@"
