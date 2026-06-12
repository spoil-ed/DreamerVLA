# DreamerVLA

DreamerVLA 是一个面向 LIBERO 的单机多 GPU 研究框架，用于 VLA 监督微调、世界模型训练和 Dreamer 风格策略优化。

```text
LIBERO HDF5
  -> no-op 过滤和 reward 标注
  -> RynnVLA / OpenVLA-OFT action-hidden sidecar
  -> DINO-style chunk world model
  -> DreamerVLA actor-critic 或 WMPO outcome
  -> LIBERO rollout 评估
```

## 快速开始

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=/path/to/dvla_data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh
TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh
CONFIG=vla_rynnvla_action_head bash scripts/train_vla.sh task=libero_goal
```

## 复现路线

1. `scripts/install_env.sh` 一键安装环境，也可以单独运行 `scripts/install/*.sh`。
2. `scripts/download_assets.sh` 一键下载权重和 LIBERO 数据，也可以单独运行 `scripts/download/*.sh`。
3. `scripts/preprocess/prepare_libero_data.sh` 生成过滤 HDF5、reward、manifest 和 sidecar。
4. `scripts/train_vla.sh` 训练 VLA。
5. `scripts/train_wm.sh` 训练 chunk world model。
6. `scripts/train_dreamervla.sh` 训练 DreamerVLA。
7. `scripts/eval_libero_vla.sh` 评估。

## 仓库结构

```text
dreamer_vla/        Python 包：runner、model、dataset、algorithm、env
configs/            Hydra route 和 LIBERO task 配置
scripts/            install、download、preprocess、train、eval 的 shell 入口
tests/              单元测试和 smoke 测试
third_party/        editable upstream dependencies
data/               未设置 DVLA_DATA_ROOT 时使用的相对数据目录
docs/               setup 和数据布局说明
```

`DVLA_DATA_ROOT` 和 `DVLA_ROOT` 相互独立；数据可以放在仓库之外的磁盘或共享存储。

完整流程见 [SETUP.md](SETUP.md)，路径约定见 [docs/data_layout.md](docs/data_layout.md)。
