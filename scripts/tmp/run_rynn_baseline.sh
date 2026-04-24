#!/bin/bash
# RynnVLA-002 official baseline smoke test (nopretokenize, libero_goal, pretrained libero_10 init)
# Uses GPU 0,1,2,3. Pair with run_dreamer_coef10.sh on GPU 4,5,6,7.
# Manual stop: Ctrl+C when loss trajectory is clear.
set -euo pipefail

cd /home/user01/yuxinglei/workspace/RynnVLA-002/rynnvla-002/exps_nopretokenize

source /home/user01/miniconda3/etc/profile.d/conda.sh
conda activate rynnvla002

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash libero_goal_baseline_pretrained_4gpu_3ep.sh 1 4
