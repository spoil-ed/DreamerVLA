# DreamerVLA

DreamerVLA 是面向 LIBERO 的单机多 GPU 训练框架，用于 rollout 采集、world
model warmup、success classifier warmup 和 OpenVLA-OFT cotrain。

```text
LIBERO rollouts
  -> reward + hidden-token HDF5 shards
  -> world model + success classifier warmup
  -> OpenVLA-OFT cotrain
  -> LIBERO rollout eval
```

OpenVLA-OFT 主线的 world-model 观测固定为当前帧 projected
`hidden_token [256,4096]`。动作解码器内部的 action slots 不写入
观测 sidecar。

## 快速开始

固定的 8xH100 Docker 复现流程见
[Docker 复现文档](docs/docker_reproduction.md)：

```bash
docker pull spoil/dreamervla:cu124-h100-v1
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  -v "$PWD/dreamervla-data:/data" spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/01_prepare_assets.sh
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  -v "$PWD/dreamervla-data:/data" spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/02_train_dreamer.sh
```

镜像包含源码和固定 revision 的完整 third_party 环境；权重、数据和输出保存在
挂载的 `/data` 中。训练中断后重新执行第二条命令会自动续训。

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm-run/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier-run/checkpoints/latest.ckpt

bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/cotrain-run
```

两个独立的官方数据上限训练入口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/world_model_training/train.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/classifier_training/train.sh
```

## 复现路线

1. `scripts/install_env.sh` 安装环境。
2. `scripts/download_assets.sh` 下载 OpenVLA-OFT one-trajectory checkpoint 和
   LIBERO 数据。
3. 使用 `scripts/experiments/cotrain/train.sh --config
   openvla_libero` 训练，并显式传入 warmup WM 和 classifier
   checkpoint。
4. 使用 `scripts/experiments/cotrain/eval.sh` 评估显式指定的 policy
   checkpoint。

## 仓库结构

```text
dreamervla/        Python 包：runner、model、dataset、algorithm、env
configs/            Hydra recipe 和 LIBERO task 配置
scripts/            install、download、preprocess、train、eval 的 shell 入口
tests/              单元测试和 smoke 测试
third_party/        被 ignore 的只读上游运行时依赖
data/               未设置 DVLA_DATA_ROOT 时使用的相对数据目录
docs/               文档索引、参考、教程、报告和论文草稿
```

`DVLA_DATA_ROOT` 和 `DVLA_ROOT` 相互独立；数据可以放在仓库之外的磁盘或共享存储。
训练产物统一写入 `outputs/<实验名>/<时间戳>/`。其中平铺的 `checkpoints/`
只包含 `latest.ckpt` 和可选的 `epoch=<epoch>-<metric>=<value>.ckpt` top-k
文件；显式开启 HF 导出后，才会创建同级 `checkpoint_hf/`。评估产物写入
`outputs/eval/<任务名>/`，输入可以是具体 checkpoint、`checkpoints/` 或训练 run root。
实验 shell 不保存训练或评估默认参数；入口和默认值由 `configs/experiment/` 中的
Hydra 配置提供，修改时使用 `key=value` override。`configs/scripts/` 保留
install、download、preprocess 和 reproduce 工作流。
缩短预算时使用 `profile=debug` 或 `profile=smoke`；Runner 不会在运行时改写
生产配置。冻结 WM/CLS 的 imagined-RL 支持路线仍为 `--config openvla_libero`。

在能够读取 GPU 共享运行目录且可以联网的 CPU 节点上，使用 W&B 官方命令持续上传
活跃的 offline run：

```bash
wandb login
wandb beta sync --live /path/to/run_root/wandb
```

应在 GPU 进程创建 `wandb/offline-run-*` 后启动，并使用 W&B 0.24.1 或更新版本。
异常退出恢复和旧目录说明见实验教程。

完整流程见 [SETUP.md](SETUP.md)，路径约定见 [docs/data_layout.md](docs/data_layout.md)。
