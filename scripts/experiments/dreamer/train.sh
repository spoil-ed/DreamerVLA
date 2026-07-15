#!/usr/bin/env bash
# Train or resume frozen-WM/CLS latent-imagination RL through the shared launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.launchers.cotrain \
  --config openvla_libero \
  "$@"
