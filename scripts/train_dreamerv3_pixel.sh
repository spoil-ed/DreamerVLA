#!/usr/bin/env bash
# Backward-compatible wrapper for DreamerV3 pixel world-model training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-dreamerv3_pixel}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
