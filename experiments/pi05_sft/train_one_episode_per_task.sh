#!/usr/bin/env bash
set -euo pipefail

# micro-batch 32 x accumulation 1 = effective batch 32/GPU, 256 globally.
# The active DreamerVLA environment owns Python and all π0.5 dependencies.
exec python -m torch.distributed.run \
  --nnodes=1 \
  --node-rank=0 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  --nproc-per-node=8 \
  -m dreamervla.train \
  experiment=pi05_libero_sft_one_episode_per_task \
  "$@"
