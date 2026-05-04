#!/usr/bin/env bash
# Backward-compatible wrapper for pretokenized/TSSM world-model training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-pretokenize}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
