#!/bin/bash
# DreamerVLA smoke test with all optim/loss hyperparams aligned to RynnVLA-002 official recipe.
# Changes vs default DreamerVLA config:
#   - vla_action_loss_coef: 1 -> 10     (loss weight to match rynn loss_ct_weights=10)
#   - vla_token_loss_coef:     -> 1     (explicit, matches rynn c_loss weight)
#   - optim.vla.name: adam -> adamw     (rynn uses AdamW with decoupled wd)
#   - optim.vla.lr: 1e-4 -> 5e-6        (rynn nopretokenize libero_goal uses 5e-6)
#   - optim.vla.betas: (0.9,0.95) -> (0.9,0.999)   (torch AdamW default, what rynn uses)
#   - optim.vla.weight_decay: 0.01 -> 0.1          (rynn uses 0.1)
#   - training.lr_warmup_steps: 500 -> 50          (rynn warmup 1% of epoch ~ 30 steps)
#
# Also edited src/models/encoder/rynnvla_encoder.py to use att_mask=True (custom block mask)
# matching rynn's generate_att_mask_3, instead of the previous att_mask=False (plain causal).
#
# Uses GPU 0,1,2,3 (complementary to dreamer-coef10 running on 4,5,6,7).
# Manual stop: Ctrl+C when loss trajectory is clear.
set -euo pipefail

cd /home/user01/yuxinglei/workspace/DreamerVLA

source /home/user01/miniconda3/etc/profile.d/conda.sh
conda activate wmpo

export CUDA_VISIBLE_DEVICES=4,5,6,7

OUT_DIR="/home/user01/yuxinglei/workspace/DreamerVLA/data/outputs/pretokenize_vla/smoke_rynn_aligned_$(date +%Y%m%d_%H%M%S)"
echo "OUT_DIR=${OUT_DIR}"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --master_port=29502 \
    scripts/train.py \
    --config-name pretokenize_vla_libero_10 \
    +training.vla_action_loss_coef=10 \
    +training.vla_token_loss_coef=1 \
    training.lr_warmup_steps=50 \
    optim.vla.name=adamw \
    optim.vla.lr=5.0e-6 \
    optim.vla.betas=[0.9,0.999] \
    optim.vla.weight_decay=0.1 \
    training.out_dir="${OUT_DIR}" \
    training.resume=false
