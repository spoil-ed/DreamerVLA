# DreamerVLA

[English](README.md)

本文说明如何复现发布的 `libero_goal` baseline。推荐使用 Docker，因为镜像中已经
包含 DreamerVLA 源码、Python/CUDA 环境和固定版本的 `third_party` 仓库。

流程固定按以下顺序运行：

1. 下载并检查 OpenVLA-OFT 权重和 LIBERO 数据，然后完成预处理。
2. 训练 WM: 30 epochs，保存 loss 最低的 checkpoint。
3. 训练 CLS: 8 epochs，保存验证集 F1 最高的 checkpoint。
4. 冻结 WM 和 CLS，再训练 Dreamer: 20,000 steps。

## 运行要求

完整发布配置需要：

- Ubuntu、8 张 NVIDIA H100 80 GB GPU，以及至少 300 GiB 可用磁盘空间。
- 准备阶段能够访问 Docker Hub、GitHub 和 Hugging Face。
- Docker 方案：安装 Docker 和 NVIDIA Container Toolkit。
- 非 Docker 方案：安装 Conda，并使用兼容 CUDA 12.4 的 NVIDIA 驱动。

## 方案 A：Docker（推荐）

### 1. 拉取镜像

```bash
docker pull spoil/dreamervla:cu124-h100-v1
mkdir -p dreamervla-data
```

### 2. 下载、检查并预处理数据和权重

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/01_prepare_assets.sh
```

### 3. 训练 WM、CLS 和 Dreamer

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/02_train_dreamer.sh
```

当前终端会直接显示日志。也可以在另一个终端使用 `docker ps`、
`docker logs -f <container-id>` 或 `docker stop <container-id>`。由于 `/data`
来自宿主机挂载目录，停止或删除容器不会删除 checkpoint。

## 方案 B：不使用 Docker

### 1. 克隆源码并设置数据目录

```bash
git clone https://github.com/spoil-ed/DreamerVLA.git
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="$DVLA_ROOT/dreamervla-data"
mkdir -p "$DVLA_DATA_ROOT"
```

后续如果打开新终端，需要重新设置 `DVLA_ROOT` 和 `DVLA_DATA_ROOT`。

### 2. 安装并检查完整环境

```bash
bash scripts/install_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dreamervla
bash scripts/install/60_verify.sh
```

安装脚本会创建 Python 3.11 环境，并安装固定版本的 OpenVLA-OFT fork 和全部
`third_party` 依赖。

### 3. 下载、检查并预处理数据和权重

```bash
bash scripts/reproduce/01_prepare_assets.sh
```

### 4. 训练 WM、CLS 和 Dreamer

```bash
bash scripts/reproduce/02_train_dreamer.sh
```

使用已经训练好的 WM/CLS 启动独立的 20-step 激进 Dreamer 实验：

```bash
bash scripts/reproduce/02_train_dreamer.sh \
  --config reproduce/train_dreamer_aggressive \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

这条命令不会重新训练 WM/CLS。它从成功与失败的全部 replay 轨迹起点生成
imagined rollout，每个 global step 都做一次 resident eval。W&B 会记录
`eval/wm_trajectory_cosine`、`eval/cls_trajectory_f1` 和
`eval/cls_trajectory_accuracy`。运行状态和输出分别保存在独立的
`training_state_aggressive.json` 与
`outputs/reproduction/libero_goal/openvla_libero_aggressive/`，不会续接原始实验。

## 中断后续训

再次执行同一条训练命令：

```bash
bash scripts/reproduce/02_train_dreamer.sh
```

Docker 方案则使用同一个 `dreamervla-data` 挂载目录，再次执行方案 A 的训练命令。
工作流会从 `checkpoints/latest.ckpt` 自动续训未完成阶段，并在检查后跳过已经完成的
阶段。

## 输出位置

数据都保存在 Docker 镜像之外。宿主机上的结果目录为：

```text
dreamervla-data/outputs/reproduction/libero_goal/world_model/
dreamervla-data/outputs/reproduction/libero_goal/classifier/
dreamervla-data/outputs/reproduction/libero_goal/dreamer/
```

技术上可以把权重复制进 Docker 镜像，但本镜像有意将权重、数据集和输出放在挂载的
数据目录中。这样镜像更小，而且下载检查结果、checkpoint 和续训状态不会随容器删除。

## 直接入口与 W&B

复现脚本内部调用公开 cotrain 入口 `scripts/experiments/cotrain/train.sh`，评估入口为
`scripts/experiments/cotrain/eval.sh`。完整参数见
[`scripts/README.md`](scripts/README.md)。

训练默认使用离线 W&B。在能够读取数据目录的联网机器上，可实时同步 Dreamer 任务：

```bash
wandb beta sync --live dreamervla-data/outputs/reproduction/libero_goal/dreamer/wandb
```

固定版本、产物检查和排错说明见
[Docker 复现细节](docs/docker_reproduction.md)，所有运行目录见
[数据布局](docs/data_layout.md)。
