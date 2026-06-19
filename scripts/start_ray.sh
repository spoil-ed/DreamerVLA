#!/usr/bin/env bash
set -euo pipefail

conda activate dreamervla

ray start --head \
  --node-ip-address="${RAY_NODE_IP:-127.0.0.1}" \
  --port="${RAY_PORT:-6379}" \
  --num-cpus="${RAY_NUM_CPUS:-1}" \
  --num-gpus="${RAY_NUM_GPUS:-0}"
