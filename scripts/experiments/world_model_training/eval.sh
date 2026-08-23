#!/usr/bin/env bash
# Evaluate DINO token one-step prediction against persistence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.diagnostics.eval_dino_token_wm "$@"
