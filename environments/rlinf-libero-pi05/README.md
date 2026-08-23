# RLinf LIBERO + pi0.5 uv profile

This is the JFS-resident Python half of the environment. It is derived from
RLinf's `embodied` extra and the `openpi` + `libero` branches in
`requirements/install.sh`; the local project is named `dreamervla`.
The repository-root `pyproject.toml` records the exact top-level runtime and
checkpoint revisions; this profile's `uv.lock` owns the complete package graph.

The groups are intentionally separate:

- `embodied`: RLinf's base embodied dependency group.
- `libero`: RLinf LIBERO and the common simulator requirements.
- `pi05`: the RLinf OpenPI package and its JAX/Orbax compatibility pins.

Inside `rlinf-sysm`, with this repository mounted at `/workspace/DreamerVLA`,
the sibling RLinf checkout mounted at `/workspace/RLinf`, and the JFS runtime
mounted at `/runtime`, mount the shared Hugging Face checkpoint root at
`/checkpoints`, then load the checked-in non-secret runtime variables with
`--env-file environments/rlinf-libero-pi05/runtime.env`:

```bash
-v /jfs/oss-import/simate_pretrain_checkpoints:/checkpoints:ro
```

```bash
cd /workspace/DreamerVLA/environments/rlinf-libero-pi05
UV_PROJECT_ENVIRONMENT=/runtime/venvs/openpi-libero \
UV_CACHE_DIR=/runtime/cache \
  /runtime/bin/uv sync --all-extras --locked \
  --python /runtime/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
```

`uv sync --all-extras` is the uv spelling of “install every optional dependency
group”. To select the target explicitly, use:

```bash
UV_PROJECT_ENVIRONMENT=/runtime/venvs/openpi-libero \
UV_CACHE_DIR=/runtime/cache \
  /runtime/bin/uv sync --extra embodied --extra libero --extra pi05 --locked \
  --python /runtime/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
```

LIBERO assets, the OpenPI tokenizer, model checkpoints, uv caches, Python, and
the virtual environment belong under `/runtime` (the JFS mount), never in the
Docker image. The verified Torch wheel used for the current environment is
retained at `../../.rlinf-runtime/downloads/`; future locked syncs use RLinf's
public CUDA 12.8 wheel mirror and cache downloads under `/runtime/cache`.

The `pi05` group contains the same generic CUDA 12 / Torch 2.11 FlashAttention
wheel selected by RLinf's official installer. It works on the usual submit
server architectures but does not contain an `sm_120` kernel for the local RTX
5090. Do not rebuild and replace it with a 5090-specific wheel when the final
job runs on a different server GPU.

The environment does not bake model weights into either layer. Put the
checkpoints below the mounted JFS root using their complete Hugging Face repo
IDs. Set the selected RLinf experiment's rollout/actor model paths to the SFT
directory at submit time.

## DreamerVLA RLinf-migrated π0.5 training

DreamerVLA keeps its own runner lifecycle while using the locally migrated RLinf
OpenPI loader, LIBERO transforms, and FSDP SFT step. The runtime paths above
resolve the training inputs to:

```text
/checkpoints/lerobot/pi05_base
/runtime/data/datasets/lerobot/physical-intelligence/libero
/checkpoints/RLinf/RLinf-Pi05-LIBERO-SFT
```

All three paths are used by the default SFT recipe: base weights come from the
first, samples from the second, and official LIBERO normalization statistics
from the third. Run eight-GPU SFT from the DreamerVLA root with:

```bash
torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=pi05_libero_sft
```

Before training or LIBERO evaluation, verify that the active interpreter still
matches the locked runtime:

```bash
python -m dreamervla.diagnostics.verify_pi05_runtime
```

The composed recipe uses `Pi05Policy`, freezes the PaliGemma VLM, trains the
Gemma expert/action projections under FSDP, accumulates to global batch 128, and writes a
trainable-parameter delta to the normal DreamerVLA `checkpoints/latest.ckpt`.

The pinned inputs are large and remain on JFS:

- `lerobot/pi05_base@b211f3d...`: about 13.5 GiB.
- `physical-intelligence/libero@a4336d5...`: about 32.5 GiB.
- optional released `RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc...`: about 7.0 GiB.
