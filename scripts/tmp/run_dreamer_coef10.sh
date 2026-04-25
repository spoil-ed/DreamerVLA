#!/bin/bash
# DreamerVLA smoke test with vla_action_loss_coef=10 (matching RynnVLA-002 official recipe).
# Uses GPU 4,5,6,7. Pair with run_rynn_baseline.sh on GPU 0,1,2,3.
# Manual stop: Ctrl+C when loss trajectory is clear.
set -euo pipefail

cd /home/user01/liops/workspace/DreamerVLA

source /home/user01/miniconda3/etc/profile.d/conda.sh
conda activate wmpo

export CUDA_VISIBLE_DEVICES=4,5,6,7

OUT_DIR="/home/user01/liops/workspace/DreamerVLA/data/outputs/pretokenize_vla/smoke_coef10_$(date +%Y%m%d_%H%M%S)"
echo "OUT_DIR=${OUT_DIR}"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --master_port=29501 \
    scripts/train.py \
    --config-name pretokenize_vla_libero_10 \
    +training.vla_action_loss_coef=10 \
    +training.vla_token_loss_coef=1 \
    training.out_dir="${OUT_DIR}" \
    training.resume=false
