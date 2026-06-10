# DreamerVLA Setup

本文只保留从新机器到 LIBERO 训练/评估的正式路径。安装、下载、环境变量和 LIBERO config 都由脚本处理，不需要每次手动 export 临时全局变量。

所有命令默认在仓库根目录执行：

```bash
cd /path/to/DreamerVLA
```

## 1. 环境

正式安装入口：

```bash
bash scripts/install_env.sh
```

脚本按固定顺序执行：

```text
apt 系统工具
  -> conda dreamervla / Python 3.11
  -> uv
  -> PyTorch 2.5.1 cu124 + requirements.txt
  -> flash-attn wheel
  -> third_party clone + editable install
  -> LIBERO editable install + repo-local config
  -> egl_probe
  -> import / CUDA 验证
```

`scripts/common_env.sh` 会被正式 shell 入口自动 source，统一设置：

```text
DVLA_ROOT, PROJECT_ROOT, PYTHON, PYTHONPATH,
LIBERO_CONFIG_PATH, MUJOCO_GL, PYOPENGL_PLATFORM,
TOKENIZERS_PARALLELISM, PYTORCH_CUDA_ALLOC_CONF
```

默认 Python 指向 `dreamervla` conda 环境。若环境在非默认位置，可覆盖：

```bash
CONDA_ENV_BIN=/abs/path/to/env/bin bash scripts/train_wm.sh
```

## 2. 权重和数据下载

下载 Hugging Face 权重与 LIBERO 数据：

```bash
bash scripts/download_assets.sh
```

常用覆盖：

```bash
LIBERO_SUITES="libero_goal libero_object" bash scripts/download_assets.sh
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 LIBERO_SUITES=libero_spatial bash scripts/download_assets.sh
```

CALVIN 默认不下载；需要时显式开启：

```bash
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=0 DOWNLOAD_CALVIN=1 \
CALVIN_TASKS=task_ABCD_D \
bash scripts/download_assets.sh
```

## 3. LIBERO 数据处理

正式一键入口：

```bash
TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh
```

默认生成当前训练配置需要的格式：

```text
data/processed_data/${TASK}_marked_t_256
data/processed_data/${TASK}_no_noops_t_256
data/processed_data/${TASK}_no_noops_t_256_pi06_remaining_reward
data/processed_data/${TASK}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
data/configs/${TASK}/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml
```

说明：

- no-op 第一步先标记到 `noop_mask`，默认 `FILTER_NOOPS=1` 再筛成现有 `*_no_noops_t_256` 路径。
- VLA SFT pretokenize 默认是 `his=1`、`len_action=1`、third view + wrist + state、256 分辨率。
- action-hidden sidecar 是另一条数据，默认 `history=2`、state、rotate180、legacy action-query hidden；`libero_goal/object` 的 `time_horizon=5`，`libero_spatial/libero_10` 的 `time_horizon=10`。

常用覆盖：

```bash
TASK=libero_10 ACTION_HIDDEN_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/preprocess/prepare_libero_data.sh

TASK=libero_goal RUN_ACTION_HIDDEN=0 bash scripts/preprocess/prepare_libero_data.sh
```

## 4. 训练

训练只选择一个正式 Hydra route config，然后用 `task=...` 或 trailing override 改少量运行参数。

### VLA SFT

```bash
CONFIG=vla_rynnvla_action_head NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_vla.sh task=libero_goal
```

切任务：

```bash
bash scripts/train_vla.sh task=libero_object
```

### One-trajectory VLA（两种方案）

单轨迹 VLA 不需要额外数据下载，复用第 2/3 节的产物。

**方案 A：自己训练**

RynnVLA 路线（产物可接主链 action-hidden → WM → DreamerVLA）：

```bash
CONFIG=vla_sft_one_trajectory NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_vla.sh task=libero_goal

# 换轨迹：选全局第 k 条 demo
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_object dataset.trajectory_offset=2
```

OpenVLA-OFT 路线（action-token SFT，`policy.use_l1_regression=false`）：

```bash
CONFIG=openvla_oft_hdf5_one_trajectory bash scripts/train_vla.sh task=libero_10

# 换 demo：dataset.demo_selection_seed=...
```

**方案 B：下载现成 OpenVLA-OFT one-traj 权重（Haozhan72）**

```bash
# 国内加速可选：export HF_ENDPOINT=https://hf-mirror.com
for name in libero-spatial libero-object libero-goal libero10; do
  hf download "Haozhan72/Openvla-oft-SFT-${name}-traj1" \
    --local-dir "data/ckpts/Openvla-oft-SFT-traj1/Openvla-oft-SFT-${name}-traj1"
done
```

也可以 `git lfs install` 后 `git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-<suite>-traj1`，放进同一目录。

目录约定 `data/ckpts/Openvla-oft-SFT-traj1/<repo 名>`，与 `scripts/eval/launch_openvla_oft_traj1_eval_g67.sh` 的默认 `CKPT_ROOT` 一致。评估：

```bash
SUITE=libero_goal bash scripts/eval/launch_openvla_oft_traj1_eval_g67.sh

# 或单卡直接跑
python scripts/eval/eval_openvla_oft_libero.py \
  --ckpt data/ckpts/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
  --suite libero_goal --policy-mode auto
```

注意：方案 B 是 token 离散头的合并权重，用于直接评估或作 RL 起点。OFT action-hidden 抽取（`scripts/preprocess/preprocess_oft_action_hidden.py`）目前固定按 L1 头组件式 ckpt 加载（`use_l1_regression=True`），方案 B 权重不能直接接 OFT WM 链路。

### World Model

```bash
CONFIG=world_model_dinowm_chunk NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_wm.sh task=libero_goal
```

### Classifier

```bash
CONFIG=latent_classifier_libero_goal_chunk \
bash scripts/train_wm.sh
```

### DreamerVLA

```bash
CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_dreamervla.sh \
  task=libero_goal \
  init.world_model_state_ckpt=/abs/path/to/wm.ckpt \
  init.classifier_state_ckpt=/abs/path/to/classifier.ckpt
```

Hydra config 用法只需要记住两点：

- `CONFIG=<route>` 选择训练路线，例如 `vla_rynnvla_action_head`、`world_model_dinowm_chunk`、`dreamervla_rynn_dino_wm_wmpo_outcome`。
- trailing args 是 Hydra override，例如 `task=libero_object`、`training.max_steps=1`、`task.hdf5_dir=/abs/path`。

## 5. 评估

VLA checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=vla \
  eval.ckpt_path=/abs/path/to/vla.ckpt \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

Dreamer checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/dreamer.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

## 6. 验证

轻量验证：

```bash
python -m pytest tests/unit_tests -q
```

数据路径验证：

```bash
test -d data/ckpts/VLA_model_256/libero_goal
test -d data/processed_data/libero_goal_no_noops_t_256
test -d data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward
test -d data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
```

训练 smoke：

```bash
OUT_DIR=/tmp/dvla_wm_smoke CONFIG=world_model_dinowm_chunk \
bash scripts/train_wm.sh task=libero_goal training.max_steps=1 dataloader.num_workers=0
```
