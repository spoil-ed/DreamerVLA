#!/usr/bin/env bash
# Summarize a classifier training run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.diagnostics.experiment_stage_checks cls-eval "$@"
