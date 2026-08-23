#!/usr/bin/env bash
# Download OpenPI's pinned LeRobot LIBERO dataset through the configured mirror.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LIBERO_LEROBOT_REPO="${LIBERO_LEROBOT_REPO:-physical-intelligence/libero}"
LIBERO_LEROBOT_REVISION="${LIBERO_LEROBOT_REVISION:-a4336d589d589045d1c56423ffdf3b88a0e19b1f}"
LIBERO_LEROBOT_TARGET="${LIBERO_LEROBOT_TARGET:-${DVLA_DATA_ROOT}/datasets/lerobot/physical-intelligence/libero}"
cd "${DVLA_ROOT}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI ('hf'). Install the locked DreamerVLA environment first." >&2
  exit 2
fi

mkdir -p "${LIBERO_LEROBOT_TARGET}"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT="${HF_ENDPOINT}" HF_HUB_DISABLE_XET=1 \
  hf download "${LIBERO_LEROBOT_REPO}" \
  --repo-type dataset \
  --revision "${LIBERO_LEROBOT_REVISION}" \
  --local-dir "${LIBERO_LEROBOT_TARGET}"
