# Docker 复现：LIBERO Goal

公开镜像 `spoil/dreamervla:cu124-h100-v1` 固定了 DreamerVLA 源码、CUDA
12.4、Python 3.11、PyTorch 2.5.1、Ray 2.55.1，以及完整 third_party
环境。镜像不包含 OpenVLA 权重、LIBERO 数据和训练输出；这些内容统一写入
宿主机挂载的 `/data`。

## 主机要求

- Ubuntu 22.04 主机与 NVIDIA Container Toolkit。
- 8 张 NVIDIA H100 80 GB GPU。
- 兼容 CUDA 12.4 容器的 NVIDIA 驱动；发布基线为 580.95.05。
- `/data` 对应的磁盘至少保留 300 GiB 空间。
- 下载阶段能够访问 GitHub、Hugging Face 和 Docker Hub。

镜像构建时会从固定 revision 安装 LIBERO、robosuite、
robosuite-task-zoo、robomimic、mimicgen、OpenVLA-OFT、定制 Transformers
fork 和 EGL probe。运行时不需要挂载本机 `third_party`，也不要覆盖
`/opt/dreamervla`。

## 1. 拉取镜像

```bash
docker pull spoil/dreamervla:cu124-h100-v1
mkdir -p dreamervla-data
```

所有后续命令使用同一个 `dreamervla-data` 目录。删除容器不会删除数据、日志或
checkpoint。

## 2. 下载、检查与预处理

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/01_prepare_assets.sh
```

脚本下载并检查：

- `Haozhan72/Openvla-oft-SFT-libero-goal-traj1`；
- 官方 `libero_goal` HDF5 数据；
- reward HDF5 与 `hidden_token [T,256,4096]` sidecar；
- 权重文件、Git LFS 内容、third_party revision、SHA-256、demo 数量与长度。

清单写入 `/data/reproduction/manifests/assets.json`。再次执行会重新校验并复用
正确资产，不会无条件覆盖。

## 3. WM 30 + CLS 8 + Dreamer 20,000

```bash
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "$PWD/dreamervla-data:/data" \
  spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/02_train_dreamer.sh
```

固定顺序为：

1. Dreamer-WM 训练 30 epoch，按最低 loss 选择最优 checkpoint。
2. Success classifier 训练 8 epoch，按最高验证 F1 选择最优 checkpoint。
3. 将选中的 WM 和 CLS 冻结，运行 Dreamer 20,000 global steps。

输出目录：

```text
/data/outputs/reproduction/libero_goal/world_model/
/data/outputs/reproduction/libero_goal/classifier/
/data/outputs/reproduction/libero_goal/dreamer/
```

状态写入 `/data/reproduction/manifests/training_state.json`。训练中断后执行同一
命令会自动续训：完成阶段先校验再跳过；未完成阶段通过其 run root 的
`checkpoints/latest.ckpt` 恢复。脚本不会自动删除或覆盖已有 run root。

## 日志与结果

每个 run root 都保留标准的 `checkpoints/`、`logs/`、`tensorboard/`、
`wandb/`、`video/`、`diagnostics/` 和 `.hydra/`。W&B 默认离线，可在能够读取
挂载目录的联网机器上执行：

```bash
wandb beta sync --live dreamervla-data/outputs/reproduction/libero_goal/dreamer/wandb
```

## 常见失败

- GPU 数量或型号不符：该发布配置只声明支持 8xH100 80 GB。
- `/data` 空间不足：扩容或换一个挂载目录后重试。
- 资产目录存在但不完整：脚本会停止并打印目录；先将异常目录移到别处，再重试。
- third_party revision/import/EGL 检查失败：重新拉取同一镜像 digest，不要在容器内
  手工升级依赖。
- 训练进程退出：保留挂载目录，重新执行第 3 步即可自动续训。

