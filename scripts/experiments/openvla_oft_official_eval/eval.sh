#!/usr/bin/env bash
# Evaluate the official OpenVLA-OFT checkpoint with its experiment config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

exec python -m dreamervla.diagnostics.eval_openvla_oft_libero "$@"
