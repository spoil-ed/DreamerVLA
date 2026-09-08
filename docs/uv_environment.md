# Native uv environment

This environment is independent of Docker. The repository's `.venv`, managed
Python installation, and uv cache live on JFS; neither `/opt/conda` nor a Docker
image is used at runtime.

## JFS paths

On the current host, use:

```bash
export UV_PYTHON_INSTALL_DIR=/jfs/oss-import/xinglei/.local/share/uv/python
export UV_CACHE_DIR=/jfs/oss-import/xinglei/.cache/uv
export UV_LINK_MODE=copy
```

The user-level uv executable (typically `~/.local/bin/uv`) is a symlink to
`/jfs/oss-import/xinglei/.local/bin/uv`. `.python-version` pins Python 3.11.15,
and `.venv/bin/python` must resolve beneath `UV_PYTHON_INSTALL_DIR`. Copy link
mode keeps the completed virtual environment independent of cache eviction.

The ignored `third_party/` checkouts must exist at the revisions pinned in
[`scripts/install/40_third_party.sh`](../scripts/install/40_third_party.sh)
before syncing, because LIBERO assets and several simulation packages are
consumed from those read-only source trees.

## Sync a CUDA profile

The root environment resolves dependencies from `pyproject.toml` without a
checked-in lockfile. `uv sync` may create a local `uv.lock`, which Git ignores;
transitive versions can change when a new resolution is generated. The separate
π0.5 profile retains its own lockfile under `configs/environments/rlinf-libero-pi05/`.

Clear inherited proxy variables for every networked uv command. For this RTX
5090 host, sync the CUDA 13.0 profile:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    uv sync --all-extras --group cu130-train
source .venv/bin/activate
```

For the upstream-compatible H100 environment instead:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    uv sync --all-extras --group cu124-train
```

The profiles are intentionally mutually exclusive:

| Group | PyTorch stack | Attention backend | Intended GPU |
| --- | --- | --- | --- |
| `cu124-train` | Torch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1 | flash-attn 2.7.1.post1 | H100 / official reproduction |
| `cu130-train` | Torch 2.13.0, torchvision 0.28.0 | PyTorch SDPA | RTX 5090 / Blackwell |

FlashAttention 2.8.3.post1 does not claim Blackwell support, so it is not part
of `cu130-train`. The OpenVLA-OFT inference path already avoids forcing
FlashAttention; do not select `flash_attention_2` explicitly on this profile.

`--all-extras` selects the maintained `ray`, `libero`, `openvla`, and `lerobot-v3` extras.
The default `dev` dependency group supplies pytest, Ruff, and pre-commit. uv
performs an exact sync, so packages installed outside `pyproject.toml` are
removed rather than silently retained.

## Runtime variables and verification

```bash
export DVLA_ROOT=/jfs/oss-import/xinglei/DreamerVLA
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
export LIBERO_CONFIG_PATH="${DVLA_DATA_ROOT}/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv pip check
python -c 'import sys, torch; print(sys._base_executable); print(torch.__version__, torch.version.cuda)'
python -m dreamervla.diagnostics.checks.verify_install
```

`verify_install` encodes the official `cu124-train` version contract and will
therefore report expected Torch/FlashAttention version differences under
`cu130-train`. Use import, CUDA allocation, EGL rendering, and project smoke
tests to validate the Blackwell profile.
