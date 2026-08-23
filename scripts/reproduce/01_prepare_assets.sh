#!/usr/bin/env bash
# Prepare and validate the public libero_goal reproduction assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

exec python -m dreamervla.launchers.reproduce \
  --config-name reproduce/prepare_assets \
  "$@"
