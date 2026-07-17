# DreamerVLA

[中文](README.zh-CN.md)

This guide reproduces the published `libero_goal` baseline. Docker is recommended
because it already contains the DreamerVLA source code, Python/CUDA packages, and
the pinned `third_party` repositories.

The workflow always runs in this order:

1. Download and check the OpenVLA-OFT weights and LIBERO data, then preprocess them.
2. Train WM: 30 epochs and save the checkpoint with the lowest loss.
3. Train CLS: 8 epochs and save the checkpoint with the highest validation F1.
4. Freeze WM and CLS, then train Dreamer: 20,000 steps.

## Requirements

The full published profile requires:

- Ubuntu, 8 NVIDIA H100 80 GB GPUs, and at least 300 GiB of free disk space.
- Internet access to Docker Hub, GitHub, and Hugging Face during preparation.
- For Docker: Docker with the NVIDIA Container Toolkit.
- Without Docker: Conda and an NVIDIA driver compatible with CUDA 12.4.

## Option A: Docker (recommended)

### 1. Pull the image

```bash
docker pull spoil/dreamervla:cu124-h100-v1
mkdir -p dreamervla-data
```

### 2. Download, check, and preprocess assets

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/01_prepare_assets.sh
```

### 3. Train WM, CLS, and Dreamer

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/02_train_dreamer.sh
```

The command prints logs in the current terminal. From another terminal, use
`docker ps`, `docker logs -f <container-id>`, or `docker stop <container-id>`.
Stopping the container does not delete checkpoints because `/data` is mounted from
the host.

## Option B: Without Docker

### 1. Clone the source and choose the data directory

```bash
git clone https://github.com/spoil-ed/DreamerVLA.git
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="$DVLA_ROOT/dreamervla-data"
mkdir -p "$DVLA_DATA_ROOT"
```

Keep `DVLA_ROOT` and `DVLA_DATA_ROOT` set in every new terminal used for this run.

### 2. Install and verify the complete environment

```bash
bash scripts/install_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dreamervla
bash scripts/install/60_verify.sh
```

The installer creates the Python 3.11 environment and installs the pinned
OpenVLA-OFT fork and all other `third_party` dependencies.

### 3. Download, check, and preprocess assets

```bash
bash scripts/reproduce/01_prepare_assets.sh
```

### 4. Train WM, CLS, and Dreamer

```bash
bash scripts/reproduce/02_train_dreamer.sh
```

## Resume after an interruption

Run the same training command again:

```bash
bash scripts/reproduce/02_train_dreamer.sh
```

For Docker, run the Docker command from Option A again with the same
`dreamervla-data` mount. The workflow automatically resumes an unfinished stage
from `checkpoints/latest.ckpt` and checks then skips completed stages.

## Outputs

All data remains outside the Docker image. On the host, results are written to:

```text
dreamervla-data/outputs/reproduction/libero_goal/world_model/
dreamervla-data/outputs/reproduction/libero_goal/classifier/
dreamervla-data/outputs/reproduction/libero_goal/dreamer/
```

Weights could technically be copied into a Docker image, but this image deliberately
keeps weights, datasets, and outputs in the mounted data directory. This keeps the
image smaller and lets downloads, checks, checkpoints, and resume state persist
independently of a container.

## Direct entry points and W&B

The reproduction script calls the public cotrain entry point
`scripts/experiments/cotrain/train.sh`. Evaluation uses
`scripts/experiments/cotrain/eval.sh`. Their full parameters are documented in
[`scripts/README.md`](scripts/README.md).

Runs use offline W&B by default. From a networked machine that can read the data
directory, stream the Dreamer run with:

```bash
wandb beta sync --live dreamervla-data/outputs/reproduction/libero_goal/dreamer/wandb
```

For detailed troubleshooting, pinned revisions, and artifact checks, see
[Docker reproduction details](docs/docker_reproduction.md). See
[the data layout](docs/data_layout.md) for every runtime path.
