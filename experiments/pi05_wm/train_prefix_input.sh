#!/usr/bin/env bash
set -euo pipefail

exec python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m dreamervla.train \
  experiment=wm_pi05_prefix_input_train \
  runner.logger.project_name=dreamer \
  "$@"
