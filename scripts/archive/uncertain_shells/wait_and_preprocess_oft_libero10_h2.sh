#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/data/spoil/workspace/DreamerVLA}
PYTHON=${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}
WAIT_SESSION=${WAIT_SESSION:-oft_h2_67}
SOURCE_DIR=${SOURCE_DIR:-$ROOT/data/processed_data/libero_10_no_noops_t_256}
SLEEP_SECONDS=${SLEEP_SECONDS:-300}

cd "$ROOT"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(date '+%F %T')] waiting for tmux session $WAIT_SESSION to finish"
  sleep "$SLEEP_SECONDS"
done

while true; do
  if "$PYTHON" - <<'PY'
import glob
import h5py
import os
import sys

source_dir = os.environ["SOURCE_DIR"]
files = sorted(glob.glob(os.path.join(source_dir, "*.hdf5")))
if len(files) < 10:
    print(f"waiting: found {len(files)}/10 hdf5 files")
    sys.exit(1)
for path in files:
    try:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            if len(data.keys()) == 0:
                raise RuntimeError("empty data group")
    except Exception as exc:
        print(f"waiting: {os.path.basename(path)} not ready: {exc}")
        sys.exit(1)
print("libero_10 source hdf5 files are ready")
PY
  then
    break
  fi
  sleep "$SLEEP_SECONDS"
done

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
  SUITES="10" \
  MASTER_PORT="${MASTER_PORT:-29517}" \
  CHUNK_SIZE="${CHUNK_SIZE:-16}" \
  bash scripts/preprocess_oft_libero_all_h2.sh \
  > data/outputs/preprocess_oft_libero_10_h2_launcher.log 2>&1
