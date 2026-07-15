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

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_onetraj_libero_cotrain \
  --wm_ckpt /path/to/wm_warmup.ckpt \
  --cls_ckpt /path/to/classifier_warmup.ckpt

bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/manual_cotrain.ckpt
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

## 主线前冻结模型可行性测试

这条路线先用官方 LIBERO reward HDF5 和
`hidden_token [256,4096]` sidecar 分别训练 WM 与 classifier 上限，
随后彻底冻结两者，只通过想象 rollout 训练 DreamerVLA policy。第一版证明路线
明确只支持 `libero_goal`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m dreamervla.launchers.frozen_model_pre_mainline task=goal ngpu=8
```

RL 阶段不创建真实环境，也不存在 WM/CLS optimizer。最后以完全相同的 suite、
task IDs、初始状态、seed、episode 数、action steps 和最大步数，分别评估原始
one-trajectory OpenVLA-OFT 与 RL policy。只有真实 LIBERO success rate 严格提升、
policy hash 改变、至少执行一次 policy optimizer step、WM/CLS hash 全程不变，且
实际评测 checkpoint 的三类状态 hash 与训练摘要完全绑定时才通过。可用
`stage=wm|classifier|rl|eval` 分阶段续跑，或用 `dry_run=true`
查看命令。这是进入正式 cotrain 主线前的因果测试，不替代现有主线。

## 复现路线

1. `scripts/install_env.sh` 安装环境。
2. `scripts/download_assets.sh` 下载 OpenVLA-OFT one-trajectory checkpoint 和
   LIBERO 数据。
3. 使用 `scripts/experiments/cotrain/train.sh --config
   openvla_onetraj_libero_cotrain` 训练，并显式传入 warmup WM 和 classifier
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
实验 shell 不保存训练或评估默认参数；入口和默认值由 `configs/experiment/` 中的
Hydra 配置提供，修改时使用 `key=value` override。`configs/scripts/` 只保留
install、download、preprocess 三类工作流。
缩短预算时使用 `profile=debug` 或 `profile=smoke`；Runner 不会在运行时改写
生产配置。冻结 WM/CLS 的 imagined-RL 支持路线仍为 `--config openvla_libero`。

离线 W&B 运行上传到 online 只需：

```bash
bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb
```

完整流程见 [SETUP.md](SETUP.md)，路径约定见 [docs/data_layout.md](docs/data_layout.md)。
