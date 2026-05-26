#!/usr/bin/env bash
# DreamerVLA LIBERO eval — task_suite = libero_spatial
#
# Usage:
#   conda activate dreamervla
#   CUDA_VISIBLE_DEVICES=7 \
#   CKPT_PATH=/abs/path/to/epoch=NNN-train_vla_loss=X.XXX.ckpt \
#     bash scripts/evals_libero/eval_libero_spatial.sh
#
# Optional overrides via env: NUM_EPISODES, HISTORY_LENGTH, ACTION_STEPS,
# MUJOCO_GL (osmesa default; set =egl to retry GPU rendering), OUT_DIR.
set -euo pipefail
TASK_SUITE=libero_spatial \
  exec "$(dirname "$0")/_eval_runner.sh" "$@"
