#!/usr/bin/env bash
# LIBERO rollout evaluation for a saved VLA checkpoint (no training).
#
# Usage:
#   conda activate dreamervla
#   bash scripts/eval_libero_vla.sh \
#     eval.ckpt_path=data/outputs/pretokenize_vla/checkpoints/epoch=013-train_vla_loss=1.984.ckpt \
#     eval.task_suite_name=libero_goal \
#     eval.num_episodes_per_task=10
#   bash scripts/eval_libero_vla.sh \
#     eval.ckpt_path=data/outputs/dreamer_vla/.../checkpoints/epoch=008-epoch_returns_mean=3.6211.ckpt \
#     eval.ckpt_kind=dreamer \
#     eval.task_suite_name=libero_10
#
# Rollout must run on a single process (single GPU); the underlying LIBERO
# benchmark does not support sharded inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CONFIG_NAME="${CONFIG_NAME:-eval_libero_vla}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/eval_libero_vla}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${CONFIG_NAME}_${TIMESTAMP}}"

mkdir -p "${OUT_DIR}"
FULL_LOG="${OUT_DIR}/run.log"

# ── MuJoCo / OpenGL backend pinning (fixes silent SIGABRT after few episodes)
# Without these, mujoco's offscreen renderer auto-detects backend and may
# pick GLFW (needs X) or leak GPU contexts across episode boundaries until
# the driver internally abort()s with no stderr message.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [ "${MUJOCO_GL}" = "egl" ]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  # EGL_DEVICE_ID picks the EGL adapter; set to 0 because CUDA_VISIBLE_DEVICES
  # already remaps the visible GPU to logical index 0 inside this process.
  export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
fi
# faulthandler dumps a Python traceback on SIGABRT/SIGSEGV/SIGFPE so future
# crashes will at least show which Python frame was active.
export PYTHONFAULTHANDLER=1

echo "Run output dir: ${OUT_DIR}"
echo "Full log file:  ${FULL_LOG}"
echo "MUJOCO_GL=${MUJOCO_GL}  PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>}  EGL_DEVICE_ID=${EGL_DEVICE_ID:-<unset>}"

# Capture *everything* (Python prints, MuJoCo/OpenSim C-side stderr, glibc
# abort messages, torchrun) into ${OUT_DIR}/run.log AND mirror to terminal.
# Shell-level `2>&1 | tee` ensures fd2 from C extensions also lands in the
# file even when Python aborts without a traceback (SIGABRT, double free,
# etc.).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@" 2>&1 | tee "${FULL_LOG}"
