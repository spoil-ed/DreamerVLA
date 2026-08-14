#!/usr/bin/env bash
set -euo pipefail

cd /home/yuxinglei/workspace/DreamerVLA

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
export WANDB_PROJECT=dreamervla
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD="$PWD/.venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12:$PWD/.venv/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/cusparselt/lib"

exec "$PWD/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  -m dreamervla.train \
  experiment=pi05_libero_sft \
  runner.logger.project_name=dreamervla \
  runner.logger.wandb_mode=online \
  "$@"
