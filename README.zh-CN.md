# DreamerVLA

DreamerVLA 是面向 LIBERO 的单机多 GPU 训练框架，用于 rollout 采集、world
model warmup、success classifier warmup 和 OpenVLA-OFT cotrain。

```text
LIBERO rollouts
  -> reward + input-token HDF5 shards
  -> world model + success classifier warmup
  -> OpenVLA-OFT cotrain
  -> LIBERO rollout eval
```

OpenVLA-OFT 主线的 world-model 观测固定为当前帧 projected
`input_token_embedding [256,4096]`。动作解码器内部的 action slots 不写入
观测 sidecar。

## 快速开始

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
```

全量 replay 的 world-model warmup 入口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  GPU_COUNT=8 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/world_model_training/train.sh
```

## 复现路线

1. `scripts/install_env.sh` 安装环境。
2. `scripts/download_assets.sh` 下载 OpenVLA-OFT one-trajectory checkpoint 和
   LIBERO 数据。
3. `scripts/e2e_coldstart_warmup_cotrain_ray.sh` 或
   `scripts/e2e_coldstart_warmup_cotrain_noray.sh` 采集 rollout。
4. 使用采集到的 replay warmup world model 和 classifier。
5. 将 `online_rollout.total_env_steps` 设为大于 0 后继续在线 cotrain。
6. `scripts/eval_libero_vla.sh` 做 LIBERO rollout 评估。

## 仓库结构

```text
dreamervla/        Python 包：runner、model、dataset、algorithm、env
configs/            Hydra recipe 和 LIBERO task 配置
scripts/            install、download、preprocess、train、eval 的 shell 入口
tests/              单元测试和 smoke 测试
third_party/        editable upstream dependencies
data/               未设置 DVLA_DATA_ROOT 时使用的相对数据目录
docs/               文档索引、参考、教程、报告和论文草稿
```

`DVLA_DATA_ROOT` 和 `DVLA_ROOT` 相互独立；数据可以放在仓库之外的磁盘或共享存储。

完整流程见 [SETUP.md](SETUP.md)，路径约定见 [docs/data_layout.md](docs/data_layout.md)。
