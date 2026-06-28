# DreamerVLA

DreamerVLA 是一个面向 LIBERO 的单机多 GPU 研究框架，用于 VLA 监督微调、世界模型训练和 Dreamer 风格策略优化。

```text
LIBERO HDF5
  -> no-op 过滤和 reward 标注
  -> RynnVLA / OpenVLA-OFT action-hidden sidecar
  -> DINO-style chunk world model
  -> DreamerVLA actor-critic 或 LUMOS
  -> LIBERO rollout 评估
```

## 快速开始

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal
bash scripts/train_vla.sh experiment=vla_rynnvla_action_head task=libero_goal
```

## 复现路线

1. `scripts/install_env.sh` 通过 Hydra 串联安装步骤；需要单步时用 `only=[20_torch]`。
2. `scripts/download_assets.sh` 通过 Hydra 选择下载步骤；需要改资源时追加 `download.*` 或 `env.*` 覆盖。
3. `scripts/preprocess/prepare_libero_data.sh task=libero_goal` 生成过滤 HDF5、reward、manifest 和 sidecar。
4. `scripts/train_vla.sh` 训练 VLA。
5. `scripts/train_wm.sh` 训练 chunk world model。
6. `scripts/train_dreamervla.sh` 训练 DreamerVLA。
7. `scripts/eval_libero_vla.sh` 评估。

## 仓库结构

```text
dreamervla/        Python 包：runner、model、dataset、algorithm、env
configs/            Hydra route 和 LIBERO task 配置
scripts/            install、download、preprocess、train、eval 的 shell 入口
tests/              单元测试和 smoke 测试
third_party/        editable upstream dependencies
data/               未设置 DVLA_DATA_ROOT 时使用的相对数据目录
docs/               文档索引、参考、教程、报告和论文草稿
```

`DVLA_DATA_ROOT` 和 `DVLA_ROOT` 相互独立；数据可以放在仓库之外的磁盘或共享存储。

完整流程见 [SETUP.md](SETUP.md)，路径约定见 [docs/data_layout.md](docs/data_layout.md)。
