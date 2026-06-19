#!/usr/bin/env bash
set -euo pipefail

conda activate dreamervla

ray status --address="${RAY_ADDRESS:-auto}"
