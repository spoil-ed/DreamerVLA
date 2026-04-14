# Dreamer-VLA

DreamerVLA is a research prototype that combines:

- `RynnVLA` as the multimodal encoder / action prior
- a compact latent world model
- PPO-style actor updates on imagined rollouts
- LIBERO-based offline data and pre-encoding utilities

## Status

The repository currently includes:

- a runnable training scaffold
- a `RynnVLAEncoder` wrapper
- LIBERO dataset loaders
- pre-encoding scripts for single-GPU and multi-GPU preprocessing
- a preencode world-model training path

The codebase still assumes several local external assets, so the main setup work is making the paths, datasets, and checkpoints line up on your machine.

## Repository Layout

```text
DreamerVLA/
├── configs/
├── data/
├── docs/
├── LIBERO/
├── scripts/
├── src/
├── download.sh
├── install.md
└── pyproject.toml
```

Key directories:

- `configs/`: experiment configs and machine-specific external paths
- `data/`: local outputs such as downloaded checkpoints and preencode shards
- `LIBERO/`: local LIBERO checkout used by the dataloaders
- `scripts/`: runnable entrypoints such as pre-encoding
- `src/`: models, dataloaders, workspaces, and training logic

## Installation

This repo is light on pinned dependencies, so the safest path is to create a dedicated conda environment first and then install the missing runtime packages explicitly.

### 1. Create the environment

```bash
conda create -n wmpo python=3.11 -y
conda activate wmpo
```

### 2. Install PyTorch

For CUDA 12.4:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

If your cluster image already provides a matching PyTorch build, keep that instead, but make sure `torch.cuda.is_available()` is `True` inside the target environment.

### 3. Install this repository

From the project root:

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
pip install --upgrade pip setuptools wheel
pip install -e .
```

`pyproject.toml` currently only declares the minimal package dependencies:

- `hydra-core`
- `omegaconf`

You will still need extra runtime packages used by the actual training and preprocessing code.

### 4. Install runtime dependencies

At minimum, install:

```bash
pip install h5py numpy tqdm transformers==4.40.1 sentencepiece huggingface_hub
```

If you use image or dataset utilities, you may also need:

```bash
pip install pillow opencv-python einops
```

Notes:

- The local Chameleon code in this repo has been patched to work with `transformers==4.40.1` by falling back to `sdpa` instead of requiring the newer flash-attention utility import.
- If you choose to upgrade `transformers`, re-test encoder loading before launching a long multi-GPU run.

## LIBERO Setup

This repository expects LIBERO data in HDF5 format and also includes a local `LIBERO/` checkout.

### 1. Install LIBERO in editable mode

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA/LIBERO
python -m pip install --no-build-isolation -e .
```

This repo includes the packaging fix for the common editable-install issue where `pip install -e .` appears to succeed but `import libero` fails outside the repository root.

### 2. Verify the install

```bash
cd /tmp
python -c "import libero; print(libero.__path__)"
```

### 3. Confirm dataset location

The current dataset config file is [rynnvla_libero_object.yaml](/home/user01/yuxinglei/workspace/DreamerVLA/configs/rynnvla_libero_object.yaml), which points to:

```bash
/home/yuxinglei/workspace/2026nips/RynnVLA-002/LIBERO/libero/datasets
```

On a new machine, this path will usually be wrong. Update `META.raw_data_dir` in that config to your real LIBERO dataset directory before running preprocessing.

## Downloading Weights

The encoder code expects local checkpoint files under `data/ckpts/`. In particular, [rynnvla_encoder.py](/home/user01/yuxinglei/workspace/DreamerVLA/src/models/encoder/rynnvla_encoder.py) defaults to:

- `data/ckpts/starting_point`
- `data/ckpts/chameleon/base_model`
- `data/ckpts/chameleon/tokenizer/text_tokenizer.json`
- `data/ckpts/chameleon/tokenizer/vqgan.yaml`
- `data/ckpts/chameleon/tokenizer/vqgan.ckpt`

### 1. Log in to Hugging Face

```bash
huggingface-cli login
```

Or, if you prefer:

```bash
hf auth login
```

### 2. Download model assets

The repository contains a helper script at [download.sh](/home/user01/yuxinglei/workspace/DreamerVLA/download.sh). It currently downloads from:

- `Alibaba-DAMO-Academy/RynnVLA-002`

Run it from the repo root:

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
bash download.sh
```

Before running it, double-check the target directories inside `download.sh`. The script currently mixes `data/ckpts/` and `data/ckpt/`, so you may want to standardize those paths if your downstream code expects everything under `data/ckpts/`.

### 3. Verify expected files exist

At minimum, verify that the paths referenced by the encoder are present:

```bash
ls data/ckpts
ls data/ckpts/chameleon
```

If your downloaded checkpoints live somewhere else, either:

- move or symlink them into `data/ckpts/`, or
- update the defaults in [rynnvla_encoder.py](/home/user01/yuxinglei/workspace/DreamerVLA/src/models/encoder/rynnvla_encoder.py)

## Machine-Specific Paths

Two configs contain absolute paths that usually need editing on a fresh machine:

- [base.yaml](/home/user01/yuxinglei/workspace/DreamerVLA/configs/base.yaml)
- [rynnvla_libero_object.yaml](/home/user01/yuxinglei/workspace/DreamerVLA/configs/rynnvla_libero_object.yaml)

Check and update:

- `paths.rynnvla_root`
- `paths.dreamerv3_root`
- `META.raw_data_dir`

If these still point to `/home/yuxinglei/...`, preprocessing or training will fail even if the Python environment is correct.

## Preencode vs Pretokenize (Important)

`preencode` 和 `pretokenize` 是两个完全不同的流程，不要混用：

- `preencode`：先跑视觉语言编码器，离线保存连续特征（`obs_embedding / next_obs_embedding / action / action_mask / reward`），主要给 world model 训练使用。
- `pretokenize`：离线保存离散 token 序列（`input_ids / labels`），主要给 VLA 的 token-level SFT 使用。

对应的数据与训练入口也不同：

- `preencode` 数据集与训练：`src/dataloader/preencode_sft_dataset.py` + `src/workspace/preencode_sft_workspace.py`（或启用 world model 的 `pretokenize_sft_workspace.py` 分支）。
- `pretokenize` 数据集与训练：`src/dataloader/pretokenize_dataset.py` + `src/workspace/pretokenize_sft_workspace.py`（tokenized SFT 分支）。

判断规则（最实用）：

- batch 里是 `obs_embedding/...` 这类连续特征，就是 `preencode` 路径。
- batch 里是 `input_ids/labels`，就是 `pretokenize` 路径。

TSSM world model（pretokenize 命名）推荐入口：

- 配置：`configs/pretokenize_tssm_world_model_libero_10.yaml`
- 脚本：`scripts/pretokenize_tssm_world_model_fsdp.sh`
- 测试脚本：`scripts/pretokenize_tssm_world_model_fsdp_test.sh`

## Pre-encoding Workflow

The pre-encoding step computes encoder embeddings in advance for world-model training.

### Quick test

Use the small test script first:

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
GPU_ID=0 MAX_SAMPLES=64 ./scripts/preencode_test.sh
```

### Full run on one GPU

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
GPU_ID=0 ./scripts/preencode.sh
```

### Full run on 8 GPUs

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
NUM_GPUS=8 ./scripts/preencode.sh
```

Or specify exact GPU IDs:

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
GPU_IDS="0 1 2 3 4 5 6 7" ./scripts/preencode.sh
```

How the multi-GPU script works:

- it splits the dataset into `N` partitions
- launches one process per GPU
- writes outputs into `part_00`, `part_01`, ...
- merges them into a single top-level `manifest.pt`

If one rank fails, inspect:

```bash
data/preencode/.../part_00.log
```

## Known Issues

- `h5py` is required for dataset loading. If it is missing, preprocessing will fail with `ModuleNotFoundError: No module named 'h5py'`.
- The machine may expose `H100` rather than `H800`; the current scripts are fine for either as long as CUDA is visible.
- The project contains several historical absolute paths from another machine. Expect to update configs before the first successful run.
- Weight layout is not fully standardized yet. The downloader and the encoder defaults should be kept consistent.

## Minimal Sanity Checks

After setup, these are the most useful quick checks:

```bash
cd /home/user01/yuxinglei/workspace/DreamerVLA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import h5py, hydra, omegaconf, transformers; print('python deps ok')"
python -c "from src.models.encoder.rynnvla_encoder import RynnVLAEncoder; print('encoder import ok')"
```

If all three pass, your environment is usually in good shape to start debugging data paths and checkpoint paths rather than Python packaging.
