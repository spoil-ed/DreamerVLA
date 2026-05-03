# Dreamer-VLA

Dreamer-VLA 是一个结合 VLA（Vision-Language-Action）编码器与 Dreamer 风格 World Model 的机器人操控研究框架。核心思路：

- 使用 **RynnVLA** 作为多模态编码器 / 动作先验（frozen）
- 在语义表征空间训练紧凑的 **latent world model**（TSSM）
- 通过 **PPO 风格** Actor-Critic 在 imagination rollouts 中优化策略
- 基于 **LIBERO** 基准的离线数据和预处理管线

> **注意**：本仓库的环境基于 [WMPO](https://github.com/WM-PO/WMPO) 和 [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002) 修改而来，数据集也沿用 RynnVLA-002 的数据处理流程。下面的安装文档会完整覆盖从零搭建环境的全部步骤。

---

## 目录

- [训练范式总览](#训练范式总览)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
  - [1. 创建 Conda 环境](#1-创建-conda-环境)
  - [2. 安装 PyTorch](#2-安装-pytorch)
  - [3. 安装本仓库](#3-安装本仓库)
  - [4. 安装 requirements.txt 依赖](#4-安装-requirementstxt-依赖)
  - [5. 安装 egl_probe（从源码）](#5-安装-egl_probe从源码)
  - [6. 安装 flash-attn（从 wheel）](#6-安装-flash-attn从-wheel)
  - [7. 安装 ColossalAI / TensorNVMe / APEX](#7-安装-colossalai--tensornvme--apex)
  - [8. 安装 LIBERO](#8-安装-libero)
  - [9. 环境验证](#9-环境验证)
- [模型权重下载](#模型权重下载)
  - [1. 下载 Chameleon 基础权重](#1-下载-chameleon-基础权重)
  - [2. 下载 RynnVLA-002 VLA / World Model 权重](#2-下载-rynnvla-002-vla--world-model-权重)
  - [3. 验证权重文件](#3-验证权重文件)
- [数据集下载与预处理](#数据集下载与预处理)
  - [1. 下载 LIBERO 数据集](#1-下载-libero-数据集)
  - [2. 数据预处理管线](#2-数据预处理管线)
  - [3. 一键预处理](#3-一键预处理)
- [路径配置](#路径配置)
- [Preencode vs Pretokenize 说明](#preencode-vs-pretokenize-说明)
- [训练](#训练)
  - [VLA SFT 训练](#vla-sft-训练)
  - [World Model 训练](#world-model-训练)
  - [VLA + World Model 联合训练](#vla--world-model-联合训练)
  - [Dreamer-VLA 完整训练](#dreamer-vla-完整训练)
- [评估](#评估)
- [已知问题与排错](#已知问题与排错)

---

## 训练范式总览

整体流水线分 **3 个阶段**：前两阶段做独立的 SFT 初始化（可并行），第三阶段把初始化好的 VLA / WM 串到 Dreamer 范式里做 imagination rollout 训练。

```
┌──────────────────────────────── Stage 0: 数据预处理 ────────────────────────────────┐
│  scripts/prepare_data.sh                                                              │
│  LIBERO HDF5 → 去 no-op → 抽图 → Chameleon VQGAN + text tokenizer 预 tokenize        │
│  产出 {image_tokens, text_tokens, action, state}                                     │
└───────────────────────────────────────────────────────────────────────────────────────┘
                │                                               │
                ▼                                               ▼
┌──────── Stage 1: SFT 初始化 VLA ────────┐   ┌──────── Stage 2: SFT 初始化 WM ────────┐
│ Config:   pretokenize_sft_libero_10.yaml │   │ Config:   pretokenize_wm_libero_10      │
│ Workspace: PretokenizeSFTWorkspace       │   │           (及 transdreamer 变种)        │
│ 作用:     在预 tokenize 数据上微调       │   │ Workspace: PretokenizeWMWorkspace       │
│           RynnVLAEncoder 头              │   │ 作用:     VLA 冻结，预训练 TSSM         │
│                                          │   │           L = L_trans + L_reward + L_KL │
└──────────────────────────────────────────┘   └─────────────────────────────────────────┘
                │                                               │
                └───────────────────┬───────────────────────────┘
                                    ▼
┌───────────────── Stage 3: Dreamer 范式 rollout 训练 ─────────────────┐
│ Config:    dreamer_vla_libero_10.yaml                                  │
│ Workspace: DreamerVLAWorkspace                                         │
│ 算法:      src/algorithms/dreamer_vla.py::imagine_actor_critic_step    │
│                                                                        │
│ 每个 batch 交替两段：                                                 │
│   Phase-1  world_model_pretrain_step()                                 │
│            继续训 WM（transition + reward + KL）                      │
│   Phase-2  imagine_actor_critic_step_v3()                              │
│            从真实 obs 编码出初始 latent (detach)                      │
│            在 WM 的 latent 空间 imagine H=15 步                        │
│            λ-returns + twohot critic + target EMA                      │
│            reward 梯度 → action → policy                               │
│                                                                        │
│ 用 training.run_wm_phase / run_actor_critic_phase 控制开关             │
└────────────────────────────────────────────────────────────────────────┘
```

### 关键注意点

**1. Stage 3 的 VLA 是冻结的**

代码里 `DreamerVLAWorkspace.run()` 显式 `freeze_module(self.encoder)`，Phase-2 imagination 里更新的是 **WM + policy + critic**，不会回传梯度到 RynnVLA 主干。如果想在 Stage 3 继续微调 VLA 本体，需要去掉这个冻结并额外配置 VLA optimizer / param group。

**2. "rollout" = imagination rollout，不是真环境 rollout**

Phase-2 全部在 WM 的 latent 空间里做 H 步 imagination，标准 Dreamer 流程。**不会**回到 `src/env/` 下的 LIBERO 真实环境采数据。如果目标是 Dyna-style（真环境 rollout → 回填 replay buffer → 再训），当前代码形态不匹配，需要额外写 replay buffer + env stepper。

**3. 两个 SFT 是平行的，不是串行依赖**

`PretokenizeSFTWorkspace` 和 `PretokenizeWMWorkspace` 共用同一份预 tokenize 数据，可以在不同机器 / 不同卡上同时启动。Stage 3 从 `init.vla_ckpt_path` 和 WM 的 ckpt 分别加载两条产出。

**4. Actor-critic 风格**

| Workspace | 算法文件 | Critic | 归一化 | Bootstrap |
|---|---|---|---|---|
| `DreamerVLAWorkspace` | `dreamer_vla.py` | twohot symlog | percentile | target critic EMA |

按 DreamerV3 (Hafner et al., 2023, §B) 的 actor-critic 实现：twohot symlog critic + percentile-normalised advantages + Polyak target critic。

**5. LIBERO 真环境评估**

Stage 3 的 loss 不足以证明策略有效——真实成功率由 `EvalLiberoVLAWorkspace`（`configs/eval_libero_vla.yaml` / `scripts/eval_libero_vla.sh`）通过在 LIBERO 里真 rollout 得到。评估时的 prompt 格式必须与训练对齐（`his=2`，图像顺序 `[prev_third, prev_wrist, cur_third, cur_wrist]`）——这条已经在 `evaluate_libero` 里落地。

---

## 项目结构

```text
DreamerVLA/
├── configs/                        # 实验配置 (Hydra YAML)
│   ├── pretokenize_sft_libero_10.yaml
│   ├── pretokenize_wm_libero_10.yaml
│   ├── pretokenize_vla_libero_10.yaml
│   ├── dreamer_vla_libero_10.yaml
│   └── ...
├── data/                           # 运行时数据（不入 git）
│   ├── ckpts/                      # 模型权重
│   ├── configs/                    # 预处理生成的训练配置
│   ├── libero/                     # LIBERO 原始数据集
│   ├── processed_data/             # 预处理中间产物
│   └── outputs/                    # 训练输出（checkpoint, log）
├── docs/                           # 技术文档
├── LIBERO/                         # LIBERO 基准本地 checkout
├── scripts/                        # 训练 / 预处理 / 评估入口脚本（薄 wrapper）
│   ├── eval_libero.sh              # LIBERO 评估
│   ├── eval_wm.sh                  # World Model 评估
│   ├── prepare_data.sh             # 一键数据预处理
│   ├── download_hf.sh              # 权重下载脚本
│   ├── install.sh                  # 环境安装脚本
│   ├── pretokenize_train_vla.sh    # VLA 训练
│   ├── pretokenize_train_wm.sh     # World Model 训练
│   ├── train_dreamer_vla.sh        # Dreamer-VLA 训练
│   └── preprocess/                 # 各步骤预处理脚本
├── src/                            # 源代码
│   ├── algorithms/                 # Dreamer-VLA, PPO/GRPO
│   ├── cli/                        # scripts/ 对应的命令行实现
│   ├── dataloader/                 # 数据集加载器
│   ├── env/                        # LIBERO 环境封装
│   ├── models/                     # 模型定义
│   │   ├── chameleon_model/        # Chameleon 视觉语言模型
│   │   ├── encoder/                # RynnVLA 编码器封装
│   │   ├── world_model/            # TSSM World Model
│   │   ├── critic/                 # Critic 网络
│   │   └── vla_policy.py           # Actor 策略网络
│   ├── preprocess/                 # 数据预处理逻辑
│   ├── trainer/                    # 分布式训练器
│   ├── utils/                      # 工具函数
│   ├── workspace/                  # 实验 Workspace
│   └── xllmx/                      # 外部 LLM 集成模块
├── pyproject.toml                  # 包配置
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

---

## 环境配置

### 系统要求

- Linux（推荐 Ubuntu 20.04+）
- NVIDIA GPU（CUDA 12.x，推荐 H100 / H800 / A100）
- Python 3.11.x
- Conda（用于管理虚拟环境）

### 1. 创建 Conda 环境

```bash
conda create -n dreamervla python=3.11 -y
conda activate dreamervla
```

### 2. 安装 PyTorch

针对 CUDA 12.4：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

安装后验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

确保输出 `2.5.1 True N`（N 为 GPU 数量）。

### 3. 安装本仓库

```bash
cd /home/user01/liops/workspace/DreamerVLA
pip install --upgrade pip setuptools wheel
pip install -e .
```

`pyproject.toml` 仅声明了 `hydra-core` 和 `omegaconf` 两个最小依赖，完整运行时依赖在下一步安装。

### 4. 安装 requirements.txt 依赖

```bash
pip install -r requirements.txt
```

此文件中包含了绝大部分运行时依赖，包括但不限于：

| 类别 | 关键包 |
|------|--------|
| 深度学习 | `xformers==v0.0.28.post3`, `diffusers==0.33.0`, `timm==0.9.10`, `peft==0.11.0` |
| 数据处理 | `h5py`, `numpy==1.26.4`, `opencv-python`, `tensorflow==2.19.0` |
| 分布式 | `ray[default]`, `tensordict` |
| 仿真 | `mujoco`, `PyOpenGL==3.1.9` |
| 日志 | `wandb` |
| Tokenizer | `tokenizers==0.19.1`, `transformers`（另需单独指定版本，见下） |

> **注意**：本仓库中 Chameleon 模型代码已 patch 为使用 `sdpa` 替代 flash-attention 的新版 import，因此与 `transformers==4.40.1` 兼容。如需指定 transformers 版本：
> ```bash
> pip install transformers==4.40.1
> ```
> 如果升级 transformers 版本，务必在长时间多 GPU 训练之前重新测试编码器加载。

### 5. 安装 egl_probe（从源码）

`egl_probe` 在 `pip install` 时经常失败，推荐从 WMPO 仓库的本地源码安装。

**问题**：`egl_probe/CMakeLists.txt` 使用了 `cmake_minimum_required(VERSION 2.8.12)`，新版 cmake 已不再兼容 `< 3.5`。

**解决方法**：

```bash
# 如果 WMPO 仓库中有 dependencies/egl_probe 目录，则从那里安装
cd /home/user01/yuxinglei/workspace/WMPO/dependencies/egl_probe

# 修复 cmake 版本要求
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
    egl_probe/CMakeLists.txt

# 安装
python -m pip install --no-build-isolation .

# 验证
python -c "import egl_probe; print(egl_probe.__file__)"
```

如果输出 `site-packages/egl_probe` 下的路径，说明安装成功。

### 6. 安装 flash-attn（从 wheel）

**不推荐**直接 `pip install flash-attn`（编译极慢且容易失败）。推荐从 [flash-attention GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases?page=1) 下载匹配的预编译 wheel。

首先确认环境信息：

```bash
python -c "import torch; print(torch.__version__); print(torch.compiled_with_cxx11_abi())"
```

根据输出选择对应的 wheel。例如 `torch==2.5.1 + CUDA 12 + cxx11_abi=False + Python 3.11`：

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.1.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

> **提示**：如果不安装 flash-attn，本仓库的 Chameleon 代码会自动 fallback 到 `sdpa`，不影响功能但可能影响速度。

### 7. 安装 ColossalAI / TensorNVMe / APEX

如果你需要使用这些组件（WMPO 的某些训练路径需要）：

```bash
# ColossalAI
pip install colossalai

# TensorNVMe（如果 WMPO 仓库中有 dependencies/TensorNVMe）
cd /home/user01/yuxinglei/workspace/WMPO/dependencies/TensorNVMe
pip install -e .

# NVIDIA APEX（从源码编译）
cd /home/user01/yuxinglei/workspace/WMPO/dependencies
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-build-isolation .
```

### 8. 安装 LIBERO

本仓库自带一个 LIBERO 的本地 checkout（位于 `LIBERO/` 目录），已修复了上游的 editable-install 问题（[Issue #31](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/31)）。

#### 安装

```bash
cd /home/user01/liops/workspace/DreamerVLA/LIBERO
python -m pip install --no-build-isolation -e .
```

#### 验证

在非仓库目录下测试 import：

```bash
cd /tmp
python -c "import libero; print(libero.__path__)"
```

如果输出正常路径（而非 `ModuleNotFoundError`），则安装成功。

#### 系统级依赖

LIBERO 的仿真渲染需要 OpenGL 相关系统库：

```bash
sudo apt install libgl1 libopengl0 libgl1-mesa-dri libgl1-mesa-glx libosmesa6-dev libosmesa6 ffmpeg
```

#### LIBERO 配置文件

LIBERO 使用全局配置文件指定数据集路径：

```bash
cat ~/.libero/config.yaml
```

确保其中 `datasets:` 字段指向你实际的数据集目录。

### 9. 环境验证

完成上述所有步骤后，运行以下检查：

```bash
cd /home/user01/liops/workspace/DreamerVLA

# 基础环境
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# Python 依赖
python -c "import h5py, hydra, omegaconf, transformers; print('python deps ok')"

# 编码器 import
python -c "from src.models.encoder.rynnvla_encoder import RynnVLAEncoder; print('encoder import ok')"

# LIBERO
python -c "import libero; print('libero ok')"

# （可选）flash-attn
python -c "import flash_attn; print(flash_attn.__version__)"
```

如果前四项全部通过，说明环境已就绪。

---

## 模型权重下载

### 前置：登录 Hugging Face

```bash
huggingface-cli login
# 或
hf auth login
```

### 1. 下载 Chameleon 基础权重

这些权重来自 `Alibaba-DAMO-Academy/WorldVLA`，包括 tokenizer、base model 和 starting point：

```bash
CKPT_DIR=/home/user01/liops/workspace/DreamerVLA/data/ckpts

# Chameleon Tokenizer（text_tokenizer.json, vqgan.yaml, vqgan.ckpt）
hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${CKPT_DIR}/chameleon/tokenizer" \
  --include "chameleon/tokenizer/*"

# Chameleon Base Model
hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${CKPT_DIR}/chameleon/base_model" \
  --include "base_model/*"

# Starting Point Checkpoint
hf download Alibaba-DAMO-Academy/WorldVLA \
  --repo-type model \
  --local-dir "${CKPT_DIR}/starting_point" \
  --include "chameleon/starting_point/*"

# Lumina-mGPT-7B-768 Tokenizer
hf download Alpha-VLLM/Lumina-mGPT-7B-768 \
  --repo-type model \
  --local-dir "${CKPT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
```

### 2. 下载 RynnVLA-002 VLA / World Model 权重

```bash
CKPT_DIR=/home/user01/liops/workspace/DreamerVLA/data/ckpts

# VLA 模型权重（256 分辨率，libero_10）
hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${CKPT_DIR}/VLA_model_256/libero_10" \
  --include "VLA_model_256/libero_10/*"

# Action World Model 权重（512 分辨率，libero_10）
hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${CKPT_DIR}/Action_World_model_512/libero_10" \
  --include "Action_World_model_512/libero_10/*"
```

也可以直接使用仓库自带脚本（需要先确认路径正确）：

```bash
cd /home/user01/liops/workspace/DreamerVLA
bash scripts/download_hf.sh
```

### 3. 验证权重文件

确保以下目录结构存在：

```text
data/ckpts/
├── chameleon/
│   ├── base_model/
│   └── tokenizer/
│       ├── text_tokenizer.json
│       ├── vqgan.yaml
│       └── vqgan.ckpt
├── starting_point/
├── models--Alpha-VLLM--Lumina-mGPT-7B-768/
├── VLA_model_256/
│   └── libero_10/
└── Action_World_model_512/
    └── libero_10/
```

快速检查：

```bash
ls data/ckpts/chameleon/tokenizer/
ls data/ckpts/VLA_model_256/libero_10/
ls data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/
```

如果权重下载到了其他位置，可以创建软链接或修改 `configs/` 下 YAML 中的路径。

---

## 数据集下载与预处理

### 1. 下载 LIBERO 数据集

LIBERO 数据集为 HDF5 格式。推荐使用 Hugging Face 源下载：

```bash
cd /home/user01/liops/workspace/DreamerVLA/LIBERO/benchmark_scripts

# 下载全部数据集（libero_goal, libero_spatial, libero_object, libero_100）
python download_libero_datasets.py --datasets all --use-huggingface

# 或只下载特定子集
python download_libero_datasets.py --datasets libero_goal --use-huggingface
```

默认下载到 `LIBERO/libero/datasets/` 目录下。也可以通过 `--download-dir` 指定其他路径。

> **提示**：如果已经在 RynnVLA-002 仓库中下载过 LIBERO 数据集，可以直接创建软链接而无需重复下载：
> ```bash
> ln -s /home/user01/yuxinglei/workspace/RynnVLA-002/LIBERO/libero/datasets \
>       /home/user01/liops/workspace/DreamerVLA/data/libero/datasets
> ```

### 2. 数据预处理管线

DreamerVLA 沿用 RynnVLA-002 的数据预处理流程，分为以下五步：

**Step 1 — 过滤 no-op 动作**

从原始 HDF5 中移除无操作帧：

```bash
LIBERO_TASK_SUITE=libero_goal IMAGE_RESOLUTION=256 \
  bash scripts/preprocess/processed_data_no_op.sh
```

输出目录：`data/processed_data/libero_goal_no_noops_t_256/`

**Step 2 — 提取图像 / 动作 / 状态**

将 HDF5 数据拆分为独立的图像、动作和机器人状态文件：

```bash
LIBERO_TASK_SUITE=libero_goal IMAGE_RESOLUTION=256 \
  bash scripts/preprocess/processed_data_save_img_action_state_wrist.sh
```

输出目录：`data/processed_data/libero_goal_image_state_action_t_256/`

**Step 3 — 生成对话 JSON**

将图像/动作/状态组织为训练所需的对话格式（包含 `<|state|>`, `<|image|>`, `<|action|>` 等特殊 token）：

```bash
LIBERO_TASK_NAME=goal IMAGE_RESOLUTION=256 ACTION_HORIZON=10 \
  bash scripts/preprocess/processed_data_generate_convs.sh
```

输出目录：`data/processed_data/convs/`

**Step 4 — Pretokenize + 合并 manifest**

将对话 JSON 预先编码为 token 序列，并合并为单一 manifest 文件：

```bash
TASK_NAME=goal IMAGE_RESOLUTION=256 ACTION_HORIZON=10 \
  bash scripts/preprocess/processed_data_pretokenize.sh
```

输出目录：`data/processed_data/tokens/` 和 `data/processed_data/concate_tokens/`

> **EOT 补全（自动发生于 Step 4）**：当 `current_frame + ACTION_HORIZON` 越过 trajectory 末尾时，管线不会丢弃该样本，而是退化到仍可达的 `effective_horizon ∈ [1, ACTION_HORIZON]`，并生成 padding 掩码。每条 pkl 新增字段：
>
> - `wm_action_mask`：长度为 `full_horizon` 的 `list[bool]`，前 `effective_horizon` 位为 `True`，其余 `False`
> - `effective_horizon` / `full_horizon`：真实步数 / 期望步数
> - `is_eot_padded`：`effective_horizon < full_horizon` 时为 `True`
>
> 下游消费：`PretokenizeDataset.collate_fn` 把 `wm_action_mask` 合进 batch 的 `action_mask`；World Model 在 `compute_loss_dict` 里用 `action_mask` 对 action chunk 做加权平均，**padding 位不参与训练**。源码：`src/preprocess/pre_tokenize_action_state_local.py::build_wm_action_mask / derive_next_obs_from_paths`。

**Step 5 — 生成训练配置 YAML**

自动生成 pretokenize 和 nopretokenize 两种训练配置：

```bash
LIBERO_TASK_SUITE=libero_goal TASK_NAME=goal IMAGE_RESOLUTION=256 ACTION_HORIZON=10 \
  bash scripts/preprocess/prepare_train_configs.sh
```

输出目录：`data/configs/libero_goal/`

### 3. 一键预处理

上述五步可以用一条命令完成：

```bash
cd /home/user01/liops/workspace/DreamerVLA
bash scripts/prepare_data.sh
```

通过环境变量覆盖默认参数：

```bash
LIBERO_TASK_SUITE=libero_10 IMAGE_RESOLUTION=256 ACTION_HORIZON=10 TASK_NAME=10 \
  bash scripts/prepare_data.sh
```

---

## 路径配置

本仓库在配置文件中使用了绝对路径，在新机器上**必须手动修改**。需要检查和更新的文件：

| 文件 | 需更新的字段 |
|------|-------------|
| `configs/pretokenize_sft_libero_10.yaml` | `training.out_dir`, `init.vla_ckpt_path`, `encoder.*_path`, `dataset.config_path` |
| `configs/pretokenize_wm_libero_10.yaml` | 同上，外加 `world_model.pretrained_model_path` |
| `configs/dreamer_vla_libero_10.yaml` | 同上 |

通用规则：将所有 `/home/user01/liops/workspace/DreamerVLA` 替换为你的实际项目根目录即可。

---

## Preencode vs Pretokenize 说明

这是两个**完全不同**的数据流程，不要混用：

| | Preencode | Pretokenize |
|---|-----------|-------------|
| **含义** | 预先跑 VLA 编码器，保存连续特征 | 预先保存离散 token 序列 |
| **输出** | `obs_embedding`, `action`, `reward` 等连续张量 | `input_ids`, `labels` 等 token ID |
| **用途** | World Model 训练 | VLA token-level SFT 训练 |
| **数据集** | `preencode_sft_dataset.py` | `pretokenize_dataset.py` |
| **Workspace** | `preencode_sft_workspace.py` | `pretokenize_sft_workspace.py` |

**判断规则**：batch 里是 `obs_embedding` 等连续特征 → preencode；是 `input_ids/labels` → pretokenize。

---

## 训练

所有训练脚本使用 `torchrun` 进行分布式训练，配合 Hydra 读取 YAML 配置。

### VLA SFT 训练

基于 pretokenize 数据的 VLA 监督微调：

```bash
conda activate dreamervla
cd /home/user01/liops/workspace/DreamerVLA

# 默认 8 GPU
bash scripts/pretokenize_train_vla.sh

# 自定义 GPU 数量和配置
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_NAME=pretokenize_sft_libero_10 \
  bash scripts/pretokenize_train_vla.sh
```

### World Model 训练

TSSM World Model 单独训练：

```bash
# 默认 4 GPU
bash scripts/pretokenize_train_wm.sh

# 自定义
NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CONFIG_NAME=pretokenize_wm_libero_10 \
  bash scripts/pretokenize_train_wm.sh
```

### Dreamer-VLA 完整训练

包含 World Model 训练阶段和 Actor-Critic imagination 训练阶段：

```bash
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_NAME=dreamer_vla_libero_10 \
  bash scripts/train_dreamer_vla.sh
```

配置中可通过 `training.run_wm_phase` 和 `training.run_actor_critic_phase` 控制单阶段或双阶段运行。

---

## 评估

在 LIBERO 环境中对训练好的 VLA checkpoint 进行 rollout 评估：

```bash
conda activate dreamervla
cd /home/user01/liops/workspace/DreamerVLA

bash scripts/eval_libero.sh \
    --ckpt_path data/outputs/pretokenize_vla/checkpoints/epoch=005-train_vla_loss=1.234.ckpt \
    --task_suite libero_goal \
    --num_episodes 10 \
    --device cuda:0
```

---

## 已知问题与排错

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'h5py'` | `pip install h5py` |
| `ModuleNotFoundError: No module named 'libero'` | 重新按 [安装 LIBERO](#8-安装-libero) 步骤执行 editable install |
| LIBERO import 成功但数据集找不到 | 检查 `~/.libero/config.yaml` 中的 `datasets:` 路径 |
| flash-attn 编译失败 | 按 [第 6 步](#6-安装-flash-attn从-wheel) 使用预编译 wheel，或直接跳过（代码会 fallback 到 sdpa） |
| 训练启动时路径报错 | 检查 `configs/` 中的绝对路径是否已更新为当前机器路径 |
| `cmake_minimum_required` 版本报错 | 按 [第 5 步](#5-安装-egl_probe从源码) 修复 CMakeLists.txt |
| `egl_probe` 安装失败 | 确保系统已安装 `libgl1`, `libosmesa6-dev` 等 OpenGL 依赖 |
| `import torch` 报 CUDA 不可用 | 检查 NVIDIA 驱动和 CUDA 版本是否与 PyTorch build 匹配 |

### 环境兼容性说明

- 本仓库在 H100 / H800 上均测试通过，只要 `nvidia-smi` 和 `torch.cuda.is_available()` 正常即可。
- `transformers` 版本推荐 `4.40.1`，更高版本可能导致 Chameleon 模型加载异常。
- `numpy` 推荐 `1.26.4`，2.x 版本与部分依赖不兼容。
- 如果安装 `xformers` 时报版本冲突，确保 PyTorch 版本严格为 `2.5.1`。
