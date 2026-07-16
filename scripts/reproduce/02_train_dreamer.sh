#!/usr/bin/env bash
# Train WM, CLS, then the frozen-WM/CLS Dreamer route with automatic resume.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

exec python -m dreamervla.launchers.reproduce \
  --config-name reproduce/train_dreamer \
  "$@"
