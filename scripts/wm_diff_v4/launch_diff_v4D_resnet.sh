#!/usr/bin/env bash
# Diff diagnostic for ResNet L4×8192 best ckpt (epoch=4).
set +e
cd /home/user01/liops/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN=action_diff_v4D_resnet_L4u8192_e004_chunk_wm_gpu4_${TS}
OUT=/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamervla_diag/${RUN}
WM_CKPT=/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamervla_diag/wm_pretrain_legacy_v4D_resnet_L4u8192_gpu4_20260521_150244/checkpoints/epoch=004-epoch_wm_loss=137.3548.ckpt
CFG=/home/user01/liops/workspace/DreamerVLA/configs/dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4d_resnet.yaml
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
echo "WM ckpt: ${WM_CKPT}"
python -m scripts.eval_action_diff_wm_vs_sft \
  --config "$CFG" --out-dir "$OUT" --world-model-ckpt "$WM_CKPT" \
  --task-suite libero_goal --task-ids 0 --num-episodes 3 --episode-horizon 200 \
  --device cuda:0 --action-head-type legacy --policy-adapter-type identity \
  --encoder-state-ckpt "" --rssm-action-scale env \
  --step-with wm --action-strategy chunk_replay --chunk-size 5 \
  2>&1 | tee -a "$OUT/diff.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
