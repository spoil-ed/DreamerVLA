#!/usr/bin/env bash
# Frozen WM/CLS policy cotrain with base-VLA eval at step 0 and PPO-VLA eval every 10 steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${SCRIPT_DIR}/e2e_frozen_model_cotrain.sh" \
  experiment=dreamervla_frozen_models_rl_ray_eval "$@"
