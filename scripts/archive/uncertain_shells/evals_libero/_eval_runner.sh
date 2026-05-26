#!/usr/bin/env bash
# Shared LIBERO eval runner for DreamerVLA.
# Adapted from RynnVLA-002 evals_libero/*.sh, with two key environment
# changes for our setup:
#
#   * MUJOCO_GL defaults to `osmesa` (software rendering).
#     EGL crashes deterministically on this box at ep3 of every task
#     (mjr_readPixels SIGABRT inside robosuite 1.4.1 + mujoco 3.8.0).
#     OSMesa is ~30% slower but proven stable.
#   * No apt-install or libstdc++ symlink hack (those targeted PAI cloud
#     env where conda libstdc++ was older than system; our conda libstdc++
#     3.4.34 is newer than system 3.4.30, so the hack would be a downgrade).
#
# Required env vars (caller sets):
#   TASK_SUITE        e.g. libero_goal | libero_10 | libero_object | libero_spatial
#   CKPT_PATH         absolute path to the .ckpt to evaluate
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES   default: 4 (eval is single-GPU only)
#   NUM_EPISODES           default: 10
#   HISTORY_LENGTH         default: 2
#   ACTION_STEPS           default: 5 for libero_goal/object, 10 otherwise
#   VLA_INIT_CKPT          default: data/ckpts/VLA_model_256/${TASK_SUITE} if present
#   ENCODER_TIME_HORIZON   default: ACTION_STEPS
#   MUJOCO_GL              default: osmesa  (set to `egl` to retry GPU rendering)
#   OUT_DIR                default: data/outputs/eval/eval_libero_vla/<suite>_<timestamp>
#
# Anything passed as positional args is forwarded as Hydra overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${TASK_SUITE:?TASK_SUITE must be set (libero_goal|libero_10|libero_object|libero_spatial)}"
: "${CKPT_PATH:?CKPT_PATH must be set to an absolute .ckpt path}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
NUM_EPISODES="${NUM_EPISODES:-10}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
if [[ -z "${ACTION_STEPS:-}" ]]; then
  case "${TASK_SUITE}" in
    libero_goal|libero_object) ACTION_STEPS=5 ;;
    *) ACTION_STEPS=10 ;;
  esac
fi
VLA_INIT_CKPT="${VLA_INIT_CKPT:-}"
if [[ -z "${VLA_INIT_CKPT}" && -d "${PROJECT_ROOT}/data/ckpts/VLA_model_256/${TASK_SUITE}" ]]; then
  VLA_INIT_CKPT="${PROJECT_ROOT}/data/ckpts/VLA_model_256/${TASK_SUITE}"
fi
ENCODER_TIME_HORIZON="${ENCODER_TIME_HORIZON:-${ACTION_STEPS}}"

# ── GL / rendering backend ───────────────────────────────────────────────
# osmesa: software (CPU) rendering — slow but no GL driver bugs
# egl   : NVIDIA EGL — fast but mjr_readPixels crashes after a few episodes
# glfw  : needs $DISPLAY (xvfb) — not installed here
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
# Mesa DRI driver search path (inherited from RynnVLA-002 fix).
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
# When using EGL backend, robosuite reads MUJOCO_EGL_DEVICE_ID; safe to
# leave unset for osmesa.
if [ "$MUJOCO_GL" = "egl" ]; then
    export PYOPENGL_PLATFORM=egl
    # robosuite asserts MUJOCO_EGL_DEVICE_ID be in CUDA_VISIBLE_DEVICES,
    # so re-use the first visible CUDA index (e.g. "7" → 7).
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
fi
# Dump a Python frame on SIGABRT/SIGSEGV/SIGFPE so future C-level crashes
# at least leave a Python traceback behind.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/outputs/eval/eval_libero_vla/${TASK_SUITE}_${TIMESTAMP}}"
mkdir -p "$OUT_DIR"
FULL_LOG="$OUT_DIR/run.log"

echo "=== DreamerVLA LIBERO eval ==="
echo "  task_suite          = $TASK_SUITE"
echo "  ckpt_path           = $CKPT_PATH"
echo "  num_episodes        = $NUM_EPISODES per task"
echo "  history_length      = $HISTORY_LENGTH"
echo "  action_steps        = $ACTION_STEPS"
echo "  vla_init_ckpt       = ${VLA_INIT_CKPT:-<config-default>}"
echo "  encoder_time_horizon= $ENCODER_TIME_HORIZON"
echo "  out_dir             = $OUT_DIR"
echo "  full_log            = $FULL_LOG"
echo "  MUJOCO_GL           = $MUJOCO_GL"
echo "  LIBGL_DRIVERS_PATH  = $LIBGL_DRIVERS_PATH"
echo "  CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
echo

EXTRA_OVERRIDES=(
  "encoder.time_horizon=${ENCODER_TIME_HORIZON}"
)
if [[ -n "${VLA_INIT_CKPT}" ]]; then
  EXTRA_OVERRIDES+=("init.vla_ckpt_path=${VLA_INIT_CKPT}")
  EXTRA_OVERRIDES+=("encoder.model_path=${VLA_INIT_CKPT}")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  --module src.cli.train \
  --config-name eval_libero_vla \
  training.out_dir="$OUT_DIR" \
  "eval.ckpt_path=\"$CKPT_PATH\"" \
  "eval.task_suite_name=$TASK_SUITE" \
  "eval.num_episodes_per_task=$NUM_EPISODES" \
  "++eval.history_length=$HISTORY_LENGTH" \
  "eval.action_steps=$ACTION_STEPS" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@" 2>&1 | tee "$FULL_LOG"
