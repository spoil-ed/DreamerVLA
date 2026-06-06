# DreamerVLA

DreamerVLA 是一个面向 LIBERO 机器人操控任务的 VLA + Dreamer-style world model 研究框架。主线流程是：

```text
LIBERO HDF5
  -> no-op 标记 / 可选筛选和 reward 处理
  -> RynnVLA action-hidden sidecar
  -> DINO-style chunk world model
  -> DreamerVLA actor-critic / WMPO outcome
  -> LIBERO rollout eval
```

完整复现命令见 [SETUP.md](SETUP.md)。本 README 只保留项目结构、关键入口和推荐路线。

## 主线复现路线

1. 环境：`bash scripts/install_env.sh`。
2. 权重和数据：`bash scripts/download_assets.sh`。
3. 数据处理：`TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh`。
4. 训练 VLA：`CONFIG=vla_rynnvla_action_head bash scripts/train_vla.sh task=libero_goal`。
5. 训练 WM：`CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal`。
6. 训练 DreamerVLA：`CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh ...`。
7. 评估：`bash scripts/eval_libero_vla.sh eval.ckpt_path=data/outputs/.../ckpt/latest.ckpt eval.ckpt_kind=vla|dreamer`。

## 目录结构

```text
DreamerVLA/
├── dreamer_vla/        # Python 包：runner、model、dataset、algorithm、env
├── configs/            # Hydra 配置；task/libero_*.yaml 保存数据和权重路径
├── scripts/            # 训练、评估、预处理、诊断入口
├── tests/              # 单测和轻量兼容性测试
├── third_party/        # LIBERO、OpenVLA-OFT、robosuite、apex 等本地依赖
├── data/               # 权重、数据、训练输出；不应提交到 git
└── docs/               # 架构说明、实验记录、历史计划
```

## 稳定入口

| 阶段 | 命令 |
| --- | --- |
| Install | `bash scripts/install_env.sh` |
| Download | `bash scripts/download_assets.sh` |
| LIBERO data | `TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh` |
| VLA SFT | `CONFIG=vla_rynnvla_action_head bash scripts/train_vla.sh task=libero_goal` |
| one-trajectory VLA | `CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal` |
| World Model | `CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal` |
| Classifier | `CONFIG=latent_classifier_libero_goal_chunk bash scripts/train_wm.sh` |
| DreamerVLA | `CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh` |
| Eval | `bash scripts/eval_libero_vla.sh eval.ckpt_path=data/outputs/.../ckpt/latest.ckpt eval.ckpt_kind=vla` |

正式 shell 入口会自动 source `scripts/common_env.sh`。常用 override：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
NGPU=4
OUT_DIR=data/outputs/<stage>/<run_name>
RUN_TAG=my_run
```

## 关键配置

Hydra 入口在 `configs/*.yaml`，任务路径在 `configs/task/*.yaml`。

最常用字段：

- `task.vla_ckpt_path`：RynnVLA base 或 SFT 权重目录。
- `task.pretokenize_config_path`：VLA SFT manifest 配置。
- `task.hdf5_dir`：默认从 no-op 标记数据筛出的 LIBERO HDF5。
- `task.hdf5_reward_dir`：remaining-steps reward HDF5。
- `task.rynnvla_action_hidden_dir`：WM/DreamerVLA 使用的 action-hidden sidecar。
- `init.world_model_state_ckpt`：DreamerVLA 初始化 WM 的 ckpt。
- `init.classifier_state_ckpt`：WMPO outcome 路线使用的 LatentSuccessClassifier ckpt。

## 推荐 smoke tests

```bash
pytest tests/unit_tests -q

OUT_DIR=/tmp/dvla_wm_smoke CONFIG=world_model_dinowm_chunk \
bash scripts/train_wm.sh task=libero_goal training.max_steps=1 dataloader.num_workers=0

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  init.vla_ckpt_path=data/ckpts/VLA_model_256/libero_goal \
  eval.ckpt_path=data/outputs/vla/rynnvla_action_head/<run>/ckpt/latest.ckpt \
  eval.ckpt_kind=vla \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=1 \
  training.device=cuda:0
```

## 进一步阅读

- [SETUP.md](SETUP.md)：从环境到 eval 的完整复现步骤。
- [onetraj_dreamervla.md](onetraj_dreamervla.md)：one-trajectory VLA SFT 到 DreamerVLA 的配置、权重和验证选择指南。
- [configs/README.md](configs/README.md)：正式配置列表。
- [scripts/README.md](scripts/README.md)：脚本入口说明。
- [docs/install.md](docs/install.md)：安装细节和 LIBERO editable install 修复。
- [docs/repository_structure.md](docs/repository_structure.md)：仓库结构说明。
