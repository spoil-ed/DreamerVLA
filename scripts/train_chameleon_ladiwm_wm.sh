#!/usr/bin/env bash
# Backward-compatible wrapper for Chameleon/LaDiWM-style world-model training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-chameleon}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
