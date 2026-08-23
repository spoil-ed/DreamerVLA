#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required before running this install step." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[install:60_verify] checking imports in conda env=${CONDA_ENV_NAME}"
echo "[install:60_verify] verifying imports and CUDA visibility"
python -m dreamervla.diagnostics.verify_install

# OpenVLA-OFT needs moojink's transformers fork (bidirectional Llama attention).
# Vanilla transformers passes every version check (both report 4.40.1) but gives
# 0% / garbage OFT actions, so assert the fork is actually active when OFT is used.
if [[ -d "${DVLA_ROOT}/third_party/openvla-oft" ]]; then
  echo "[install:60_verify] verifying OpenVLA-OFT transformers fork (bidirectional Llama attention)"
  python - <<'PY'
import os, sys, transformers
p = os.path.join(os.path.dirname(transformers.__file__), "models", "llama", "modeling_llama.py")
src = open(p).read()
is_fork = ("is_causal=False" in src) and ("Moo Jin" in src)
print(f"[install:60_verify] transformers {transformers.__version__} @ {transformers.__file__} (fork={is_fork})")
if not is_fork:
    sys.exit(
        "[install:60_verify] FATAL: transformers is VANILLA, not the OpenVLA-OFT fork "
        "(moojink/transformers-openvla-oft). OFT inference will give 0% garbage actions. "
        "Re-run 40_third_party.sh (offline: set TRANSFORMERS_OFT_FORK_SRC). See SETUP.md section 1."
    )
PY

  # peft must stay compatible with the OFT transformers fork (4.40.1). peft>=0.12
  # imports transformers.EncoderDecoderCache, which the fork lacks, so OFT policy load
  # crashes (ImportError) inside collect/cotrain -- and only surfaces deep in a Ray
  # inference worker. requirements.txt pins peft==0.11.0; a stray openvla-oft install
  # WITHOUT --no-deps upgrades it (pyproject asks peft>=0.15). Catch it here.
  echo "[install:60_verify] verifying peft is compatible with the OpenVLA-OFT transformers fork"
  python - <<'PY'
import sys
try:
    import peft
    from peft import LoraConfig, get_peft_model  # noqa: F401  (the exact OFT import)
except Exception as e:
    print(f"[install:60_verify] peft import FAILED: {e!r}", file=sys.stderr)
    sys.exit(
        "[install:60_verify] FATAL: peft is incompatible with the OpenVLA-OFT transformers "
        "fork (4.40.1). peft>=0.12 imports transformers.EncoderDecoderCache (absent in the "
        "fork), so OFT policy load crashes in collect/cotrain (incl. Ray inference workers). "
        "Pin it back: pip install peft==0.11.0  (requirements.txt pins this; a stray "
        "openvla-oft install WITHOUT --no-deps upgrades it). See SETUP.md section 1."
    )
print(f"[install:60_verify] peft {peft.__version__} OK (compatible with OFT transformers fork)")
PY
fi
