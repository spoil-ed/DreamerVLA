#!/usr/bin/env bash
# Train DreamerVLA with the pixel DreamerV3 world model on GPUs 4,5,6,7.
#
# Default policy is `pixel_vlaactor`: a Dreamer RSSM feature adapter feeding
# the VLA ActionHead actor. Set POLICY_KIND=mlp to use the smaller Gaussian MLP
# policy from configs/dreamer_vla_libero_goal_dreamerv3_pixel_actor.yaml.
#
# Common usage:
#   bash scripts/train_dreamer_vla_pixel.sh
#   DETACH=1 bash scripts/train_dreamer_vla_pixel.sh
#   POLICY_KIND=mlp bash scripts/train_dreamer_vla_pixel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="${DREAMERVLA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}:$PATH"

POLICY_KIND="${POLICY_KIND:-pixel_vlaactor}"
case "${POLICY_KIND}" in
  mlp)
    CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_dreamerv3_pixel_actor}"
    ;;
  pixel_vlaactor|vlaactor)
    CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_dreamerv3_pixel_vlaactor}"
    ;;
  *)
    echo "Unknown POLICY_KIND=${POLICY_KIND}; use mlp or pixel_vlaactor." >&2
    exit 2
    ;;
esac

export CONFIG_NAME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export RUN_TAG="${RUN_TAG:-${POLICY_KIND}_bs1}"
export OUT_DIR_BASE="${OUT_DIR_BASE:-${PROJECT_ROOT}/data/outputs/dreamervla}"
export PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"

if [[ "${DETACH:-0}" == "1" && "${DREAMERVLA_DETACHED:-0}" != "1" ]]; then
  export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

  PREVIEW="$(DRY_RUN=1 bash scripts/train_dreamer_vla.sh "$@")"
  DETACHED_OUT_DIR="$(printf '%s\n' "${PREVIEW}" | awk -F': ' '/^Run output dir:/ {print $2; exit}')"
  if [[ -z "${DETACHED_OUT_DIR}" ]]; then
    echo "Could not infer output dir from scripts/train_dreamer_vla.sh dry run." >&2
    printf '%s\n' "${PREVIEW}" >&2
    exit 1
  fi

  mkdir -p "${DETACHED_OUT_DIR}"
  DETACHED_LOG="${LOG:-${DETACHED_OUT_DIR}/train_stdout.log}"

  echo "Launching detached pixel DreamerVLA training."
  echo "Run output dir: ${DETACHED_OUT_DIR}"
  echo "Log: ${DETACHED_LOG}"

  DETACH=0 DREAMERVLA_DETACHED=1 setsid bash "$0" "$@" > "${DETACHED_LOG}" 2>&1 < /dev/null &
  DETACHED_PID="$!"
  echo "${DETACHED_PID}" > "${DETACHED_OUT_DIR}/train.pid"
  echo "PID: ${DETACHED_PID}"
  exit 0
fi

exec bash scripts/train_dreamer_vla.sh "$@"
