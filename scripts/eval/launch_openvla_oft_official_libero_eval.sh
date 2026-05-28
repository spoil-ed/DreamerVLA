#!/usr/bin/env bash
set -euo pipefail

# Official-style OpenVLA-OFT LIBERO eval launcher for DreamerVLA.
#
# This uses the OpenVLA-OFT `experiments.robot.libero.run_libero_eval`
# internals through scripts/eval/eval_openvla_oft_libero.py, but isolates each
# LIBERO task in its own Python process. That mirrors the safer subprocess
# lifecycle used by RLinf/SimpleVLA-RL and avoids robosuite render context
# crashes when switching tasks in-process.

ROOT="${ROOT:-/mnt/data/spoil/workspace/DreamerVLA}"
PYTHON_BIN="${PYTHON_BIN:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${ROOT}/third_party/openvla-oft}"

SUITE="${SUITE:-libero_goal}"
case "${SUITE}" in
  libero10|libero_long) SUITE="libero_10" ;;
esac
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/data/ckpts/Openvla-oft-SFT-traj1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/outputs/eval/openvla_oft_official_libero}"
STAGED_CKPT_ROOT="${STAGED_CKPT_ROOT:-${ROOT}/data/tmp_ckpts/openvla_oft_official_eval}"
USE_STAGED_CKPT="${USE_STAGED_CKPT:-1}"

NUM_TRIALS="${NUM_TRIALS:-10}"
POLICY_MODE="${POLICY_MODE:-discrete}"
CAMERA_INPUTS="${CAMERA_INPUTS:-primary}"
case "${CAMERA_INPUTS}" in
  primary) DEFAULT_NUM_IMAGES=1 ;;
  primary+wrist|primary_wrist)
    CAMERA_INPUTS="primary+wrist"
    DEFAULT_NUM_IMAGES=2
    ;;
  *) echo "Unsupported CAMERA_INPUTS=${CAMERA_INPUTS}; use primary or primary+wrist" >&2; exit 2 ;;
esac
NUM_IMAGES="${NUM_IMAGES:-${DEFAULT_NUM_IMAGES}}"
if [[ "${NUM_IMAGES}" != "${DEFAULT_NUM_IMAGES}" ]]; then
  echo "CAMERA_INPUTS=${CAMERA_INPUTS} expects NUM_IMAGES=${DEFAULT_NUM_IMAGES}, got NUM_IMAGES=${NUM_IMAGES}" >&2
  exit 2
fi
USE_PROPRIO="${USE_PROPRIO:-0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
ENV_IMG_RES="${ENV_IMG_RES:-256}"
SEED="${SEED:-7}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"

GPU_A="${GPU_A:-6}"
GPU_B="${GPU_B:-7}"
TASKS_A="${TASKS_A-0 1 2 3 4}"
TASKS_B="${TASKS_B-5 6 7 8 9}"
SLEEP_BETWEEN_WORKERS="${SLEEP_BETWEEN_WORKERS:-10}"
SLEEP_BETWEEN_TASKS="${SLEEP_BETWEEN_TASKS:-3}"

case "${SUITE}" in
  libero_spatial) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-spatial-traj1}" ;;
  libero_object) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-object-traj1}" ;;
  libero_goal) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-goal-traj1}" ;;
  libero_10) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero10-traj1}" ;;
  libero_90) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero90-traj1}" ;;
  *) echo "Unsupported SUITE=${SUITE}" >&2; exit 2 ;;
esac

CKPT="${CKPT:-${CKPT_ROOT}/${CKPT_NAME}}"
CKPT_LABEL="${CKPT_LABEL:-$(basename "${CKPT}")}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${SUITE}_${CKPT_LABEL}_${TS}}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_ID}}"
SCRIPT_PATH="$(readlink -f "$0")"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export OPENVLA_OFT_ROOT

stage_checkpoint() {
  local src="$1"
  local dst="$2"

  if [[ ! -d "${src}" ]]; then
    echo "Checkpoint not found: ${src}" >&2
    exit 2
  fi

  mkdir -p "${dst}"
  find "${src}" -mindepth 1 -maxdepth 1 -type f \
    ! -name '*.safetensors' \
    ! -name '*.pt' \
    ! -name '*.back.*' \
    -exec cp -a '{}' "${dst}/" ';'
  find "${src}" -mindepth 1 -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.pt' \) \
    -exec ln -sf '{}' "${dst}/" ';'
  find "${src}" -mindepth 1 -maxdepth 1 -type d -exec cp -a '{}' "${dst}/" ';'
}

run_worker() {
  local gpu="$1"
  local task_list="$2"
  local worker_name="$3"
  local worker_log="${RUN_DIR}/${worker_name}.log"
  local exit_code=0

  mkdir -p "${RUN_DIR}"
  cd "${ROOT}"
  export CUDA_VISIBLE_DEVICES="${gpu}"

  echo "worker=${worker_name}" | tee "${worker_log}"
  echo "gpu=${gpu} suite=${SUITE} tasks=${task_list} ckpt=${MODEL_FOR_RUN:-${CKPT}}" | tee -a "${worker_log}"
  echo "MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}" | tee -a "${worker_log}"

  for tid in ${task_list}; do
    local task_dir="${RUN_DIR}/task${tid}"
    local task_status=0
    local proprio_arg="--no-use-proprio"
    local video_arg=()

    if [[ "${USE_PROPRIO}" == "1" ]]; then
      proprio_arg="--use-proprio"
    fi
    if [[ "${SAVE_VIDEOS}" == "1" ]]; then
      video_arg=(--save-videos)
    fi

    mkdir -p "${task_dir}"
    echo "starting task=${tid}" | tee -a "${worker_log}"
    set +e
    "${PYTHON_BIN}" -u scripts/eval/eval_openvla_oft_libero.py \
      --ckpt "${MODEL_FOR_RUN:-${CKPT}}" \
      --suite "${SUITE}" \
      --task-ids "${tid}" \
      --num-trials "${NUM_TRIALS}" \
      --gpu-id "${gpu}" \
      --seed "${SEED}" \
      --policy-mode "${POLICY_MODE}" \
      --camera-inputs "${CAMERA_INPUTS}" \
      --num-images "${NUM_IMAGES}" \
      ${proprio_arg} \
      --num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}" \
      --env-img-res "${ENV_IMG_RES}" \
      --output-dir "${task_dir}" \
      --openvla-oft-root "${OPENVLA_OFT_ROOT}" \
      --run-note "${RUN_ID}_task${tid}" \
      "${video_arg[@]}" \
      2>&1 | tee -a "${worker_log}"
    task_status=${PIPESTATUS[0]}
    set -e
    echo "task=${tid} exit_code=${task_status}" | tee -a "${worker_log}"
    if (( task_status != 0 )); then
      exit_code="${task_status}"
    fi
    sleep "${SLEEP_BETWEEN_TASKS}"
  done

  echo "worker=${worker_name} exit_code=${exit_code}" | tee -a "${worker_log}"
  exit "${exit_code}"
}

if [[ "${DREAMERVLA_OFFICIAL_LIBERO_WORKER:-0}" == "1" ]]; then
  if (( $# != 3 )); then
    echo "worker mode expects: <gpu> <task-list> <worker-name>" >&2
    exit 2
  fi
  run_worker "$1" "$2" "$3"
fi

mkdir -p "${RUN_DIR}" "${STAGED_CKPT_ROOT}"

MODEL_FOR_RUN="${MODEL_FOR_RUN:-${CKPT}}"
if [[ "${USE_STAGED_CKPT}" == "1" ]]; then
  MODEL_FOR_RUN="${STAGED_CKPT_ROOT}/${CKPT_LABEL}_${RUN_ID}"
  stage_checkpoint "${CKPT}" "${MODEL_FOR_RUN}"
fi

printf '%s\n' "${RUN_DIR}" | tee "${OUT_ROOT}/latest_${SUITE}_official.txt" >/dev/null

launch_worker_session() {
  local gpu="$1"
  local task_list="$2"
  local worker_suffix="$3"
  local session_name="openvla_oft_official_${SUITE}_g${gpu}_${worker_suffix}_${TS}"

  if [[ -z "${task_list// }" ]]; then
    return 0
  fi

  tmux new-session -d -s "${session_name}" \
    "cd '${ROOT}' && env DREAMERVLA_OFFICIAL_LIBERO_WORKER=1 ROOT='${ROOT}' PYTHON_BIN='${PYTHON_BIN}' OPENVLA_OFT_ROOT='${OPENVLA_OFT_ROOT}' SUITE='${SUITE}' CKPT='${CKPT}' MODEL_FOR_RUN='${MODEL_FOR_RUN}' RUN_DIR='${RUN_DIR}' RUN_ID='${RUN_ID}' NUM_TRIALS='${NUM_TRIALS}' POLICY_MODE='${POLICY_MODE}' CAMERA_INPUTS='${CAMERA_INPUTS}' NUM_IMAGES='${NUM_IMAGES}' USE_PROPRIO='${USE_PROPRIO}' NUM_OPEN_LOOP_STEPS='${NUM_OPEN_LOOP_STEPS}' ENV_IMG_RES='${ENV_IMG_RES}' SEED='${SEED}' SAVE_VIDEOS='${SAVE_VIDEOS}' MUJOCO_GL='${MUJOCO_GL}' PYOPENGL_PLATFORM='${PYOPENGL_PLATFORM}' TF_FORCE_GPU_ALLOW_GROWTH='${TF_FORCE_GPU_ALLOW_GROWTH}' TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}' SLEEP_BETWEEN_TASKS='${SLEEP_BETWEEN_TASKS}' '${SCRIPT_PATH}' '${gpu}' '${task_list}' '${session_name}'"
  echo "${session_name}" | tee -a "${RUN_DIR}/tmux_sessions.txt"
}

echo "run_dir=${RUN_DIR}"
echo "model_for_run=${MODEL_FOR_RUN}"
launch_worker_session "${GPU_A}" "${TASKS_A}" "a"
sleep "${SLEEP_BETWEEN_WORKERS}"
launch_worker_session "${GPU_B}" "${TASKS_B}" "b"
echo "tmux sessions listed in ${RUN_DIR}/tmux_sessions.txt"
