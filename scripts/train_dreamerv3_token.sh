#!/usr/bin/env bash
# Backward-compatible wrapper for DreamerV3 token world-model training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-dreamerv3_token}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
