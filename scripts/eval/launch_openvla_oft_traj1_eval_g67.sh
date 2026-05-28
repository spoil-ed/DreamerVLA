#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data/spoil/workspace/DreamerVLA}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/data/ckpts/Openvla-oft-SFT-traj1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/outputs/eval/openvla_oft_traj1_g67}"
PYTHON_BIN="${PYTHON_BIN:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${ROOT}/third_party/openvla-oft}"
SUITE="${SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-10}"
POLICY_MODE="${POLICY_MODE:-auto}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-20}"

case "${SUITE}" in
  libero_spatial) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-spatial-traj1}" ;;
  libero_object) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-object-traj1}" ;;
  libero_goal) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero-goal-traj1}" ;;
  libero_10) CKPT_NAME="${CKPT_NAME:-Openvla-oft-SFT-libero10-traj1}" ;;
  *) echo "Unsupported SUITE=${SUITE}" >&2; exit 2 ;;
esac

CKPT="${CKPT:-${CKPT_ROOT}/${CKPT_NAME}}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${SUITE}_${CKPT_NAME}_${TS}"
mkdir -p "${RUN_DIR}"

VIDEO_ARG=""
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
  VIDEO_ARG="--save-videos"
fi

launch_one() {
  local gpu="$1"
  local tasks="$2"
  local name="openvla_oft_${SUITE}_traj1_g${gpu}_${TS}"
  local out_dir="${RUN_DIR}/gpu${gpu}_tasks${tasks//,/_}"
  tmux new-session -d -s "${name}" \
    "set -o pipefail; cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' OPENVLA_OFT_ROOT='${OPENVLA_OFT_ROOT}' TF_FORCE_GPU_ALLOW_GROWTH=true MUJOCO_GL=egl PYOPENGL_PLATFORM=egl '${PYTHON_BIN}' scripts/eval/eval_openvla_oft_libero.py \
      --ckpt '${CKPT}' \
      --suite '${SUITE}' \
      --task-ids '${tasks}' \
      --num-trials '${NUM_TRIALS}' \
      --policy-mode '${POLICY_MODE}' \
      --output-dir '${out_dir}' \
      --openvla-oft-root '${OPENVLA_OFT_ROOT}' \
      ${VIDEO_ARG} \
      2>&1 | tee '${RUN_DIR}/${name}.log'; echo exit_code=\$? | tee -a '${RUN_DIR}/${name}.log'"
  echo "${name}"
}

echo "${RUN_DIR}" | tee /tmp/openvla_oft_traj1_g67_latest_outdir
launch_one 6 "0-4"
sleep "${SLEEP_BETWEEN}"
launch_one 7 "5-9"
echo "logs: ${RUN_DIR}"
