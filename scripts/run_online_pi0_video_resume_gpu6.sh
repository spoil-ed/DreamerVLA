#!/usr/bin/env bash
set -u

ROOT=/home/user01/liops/workspace/DreamerVLA
PYTHON=/home/user01/miniconda3/envs/dreamervla/bin/python
OUT_DIR="${ROOT}/data/outputs/dreamervla_online/dvl_online_pi0_h2_pi06wm11000_gpu6_20260514"
WM_CKPT="${ROOT}/data/outputs/worldmodel/rynn_backbone_dreamerv3_wm/rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed_pi0_action_hidden_vla_policy_h2_pi06_remaining_reward_resume10000_bs96_gpu456_20260514_resume_pi06/ckpt/pi06_reward_step_011000_snapshot.ckpt"
RESUME_CKPT="${OUT_DIR}/checkpoints/latest.ckpt"
LOG="${ROOT}/data/outputs/logs/dvl_online.log"

export PATH="/home/user01/miniconda3/envs/dreamervla/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=6
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-dvl-online

mkdir -p "${ROOT}/data/outputs/logs" /tmp/matplotlib-dvl-online "${OUT_DIR}/videos"
cd "${ROOT}" || exit 1

{
  echo "===== restart aligned online osmesa video resume $(date) ====="
  "${PYTHON}" -X faulthandler -u scripts/train_online_pi0_action_hidden_dreamervla.py \
    --out-dir "${OUT_DIR}" \
    --world-model-ckpt "${WM_CKPT}" \
    --resume-ckpt "${RESUME_CKPT}" \
    --task-suite libero_goal \
    --task-ids 0 \
    --episode-horizon 200 \
    --total-env-steps 200000 \
    --max-train-updates 50000 \
    --train-ratio 32 \
    --batch-size 4 \
    --min-replay 64 \
    --log-every 10 \
    --save-every 200 \
    --rssm-action-scale env \
    --video-every-env-steps 300 \
    --video-fps 30 \
    --video-max-frames 200
  code=$?
  echo "===== dvl_online exit_code=${code} $(date) ====="
  exit "${code}"
} >> "${LOG}" 2>&1
