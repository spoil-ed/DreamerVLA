#!/usr/bin/env bash
# Trainable WM/CLS policy cotrain with matched real-LIBERO VLA eval every 10 steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${SCRIPT_DIR}/e2e_frozen_model_cotrain.sh" \
  experiment=dreamervla_wmcls_cotrain_ray_eval "$@"
