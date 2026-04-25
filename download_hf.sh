#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/user01/liops/workspace/DreamerVLA/data"

mkdir -p "${ROOT_DIR}/ckpts/libero_10"
mkdir -p "${ROOT_DIR}/ckpts/chameleon/tokenizer"
mkdir -p "${ROOT_DIR}/ckpts/chameleon/base_model"
mkdir -p "${ROOT_DIR}/ckpts/starting_point"

hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/libero_10" \
  --include "VLA_model_256/libero_10/*"

hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/libero_10" \
  --include "Action_World_model_512/libero_10/*"

hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/chameleon/tokenizer" \
  --include "chameleon/tokenizer/*"

hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/chameleon/base_model" \
  --include "base_model/*"

hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/starting_point" \
  --include "chameleon/starting_point/*"

hf download Alpha-VLLM/Lumina-mGPT-7B-768 \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
