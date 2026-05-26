#!/usr/bin/env bash
# v4-C-wide BEST ckpt (epoch=004, loss=145.03) diff diagnostic.
# step_with=wm + chunk_replay matches eval_libero deployment.
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN=action_diff_v4Cwide_e004_chunk_wm_gpu4_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/${RUN}}"
WM_CKPT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/wm_pretrain_legacy_v4Cwide_u16384L2_gpu6_20260521_073631/checkpoints/epoch=004-epoch_wm_loss=145.0321.ckpt
CFG=/mnt/data/spoil/workspace/DreamerVLA/configs/dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4c_wide.yaml
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
echo "WM ckpt: ${WM_CKPT}"
python -m scripts.eval_action_diff_wm_vs_sft \
  --config "$CFG" --out-dir "$OUT" --world-model-ckpt "$WM_CKPT" \
  --task-suite libero_goal --task-ids 0 --num-episodes 1 --episode-horizon 200 \
  --device cuda:0 --action-head-type legacy --policy-adapter-type identity \
  --encoder-state-ckpt "" --rssm-action-scale env \
  --step-with wm --action-strategy chunk_replay --chunk-size 5 \
  2>&1 | tee -a "$OUT/diff.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
