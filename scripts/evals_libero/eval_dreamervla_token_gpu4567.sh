#!/usr/bin/env bash
# Parallel LIBERO_10 rollout eval for a DreamerVLA checkpoint on GPUs 4,5,6,7.
#
# Required:
#   CKPT_PATH=/abs/path/to/dreamervla.ckpt bash scripts/evals_libero/eval_dreamervla_token_gpu4567.sh
#
# Useful overrides:
#   NUM_EPISODES=1                 episodes per LIBERO task
#   GPUS=4,5,6,7                   one eval worker per GPU
#   MUJOCO_GL=egl                  egl is fast; osmesa is slower but more stable
#   DREAMER_ACTION_REPEAT=1        keep 1 for faithful single-step Dreamer actor eval
#   TASK_SHARDS="0:3,3:3,6:2,8:2" task_start:max_tasks per worker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CKPT_PATH:?CKPT_PATH must point to a DreamerVLA .ckpt}"

export PATH="${DREAMERVLA_ENV_BIN:-/home/user01/miniconda3/envs/dreamervla/bin}:$PATH"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED=1

GPUS="${GPUS:-4,5,6,7}"
NUM_EPISODES="${NUM_EPISODES:-1}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
ACTION_STEPS="${ACTION_STEPS:-10}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
DREAMER_ACTION_REPEAT="${DREAMER_ACTION_REPEAT:-1}"
TASK_SHARDS="${TASK_SHARDS:-0:3,3:3,6:2,8:2}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/data/outputs/eval/eval_libero_vla/token_gpu4567_${TIMESTAMP}}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
IFS=',' read -r -a SHARD_LIST <<< "${TASK_SHARDS}"

if [[ "${#GPU_LIST[@]}" -ne "${#SHARD_LIST[@]}" ]]; then
  echo "GPUS and TASK_SHARDS must have the same length." >&2
  echo "  GPUS=${GPUS}" >&2
  echo "  TASK_SHARDS=${TASK_SHARDS}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
echo "=== Parallel DreamerVLA token LIBERO_10 eval ==="
echo "  ckpt              = ${CKPT_PATH}"
echo "  out_root          = ${OUT_ROOT}"
echo "  gpus              = ${GPUS}"
echo "  task_shards       = ${TASK_SHARDS}"
echo "  episodes/task     = ${NUM_EPISODES}"
echo "  dreamer_repeat    = ${DREAMER_ACTION_REPEAT}"
echo "  mujoco_gl         = ${MUJOCO_GL}"

pids=()
for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  shard="${SHARD_LIST[$idx]}"
  task_start="${shard%%:*}"
  max_tasks="${shard##*:}"
  shard_out="${OUT_ROOT}/task${task_start}_n${max_tasks}_gpu${gpu}"
  mkdir -p "${shard_out}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MUJOCO_GL="${MUJOCO_GL}"
    if [[ "${MUJOCO_GL}" == "egl" ]]; then
      export PYOPENGL_PLATFORM=egl
      export MUJOCO_EGL_DEVICE_ID="${gpu}"
      export EGL_DEVICE_ID=0
    fi
    export CKPT_PATH
    export NUM_EPISODES
    export HISTORY_LENGTH
    export ACTION_STEPS
    export OUT_DIR="${shard_out}"
    bash scripts/evals_libero/eval_libero_10.sh \
      eval.ckpt_kind=dreamer \
      "eval.task_start=${task_start}" \
      "eval.max_tasks=${max_tasks}" \
      "eval.dreamer_action_repeat=${DREAMER_ACTION_REPEAT}"
  ) &
  pids+=("$!")
  echo "  launched gpu=${gpu} task_start=${task_start} max_tasks=${max_tasks} pid=${pids[-1]} out=${shard_out}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

python - <<'PY' "${OUT_ROOT}" || true
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
episodes = 0
successes = 0
print("\n=== shard metrics ===")
for path in sorted(root.glob("*/eval_libero_metrics.json")):
    data = json.loads(path.read_text())
    ep = int(data.get("eval_total_episodes", 0))
    ok = int(data.get("eval_total_successes", 0))
    episodes += ep
    successes += ok
    print(f"{path.parent.name}: {ok}/{ep} = {ok / ep if ep else 0:.1%}")
print(f"overall: {successes}/{episodes} = {successes / episodes if episodes else 0:.1%}")
PY

exit "${status}"
