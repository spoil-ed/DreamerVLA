# Dreamer-VLA

Dreamer-VLA 是一个结合 VLA（Vision-Language-Action）编码器与 Dreamer 风格 World Model 的机器人操控研究框架。当前主线是 **pi0 action-hidden DreamerV3 WM**：

- 复用 **RynnVLA / Chameleon backbone + pi0 action-query block**，一直抽到 action hidden
- action hidden 同时作为 VLA action head 输入，以及 DreamerV3 RSSM 的 observation embedding
- 第一版固定共享 VLA backbone，预计算 action-hidden sidecar，再训练 DreamerV3 RSSM / decoder / reward / continue
- joint finetune 与 DreamerVLA actor-critic 训练暂列为 follow-up，避免破坏已经可用的 VLA 表达
- 基于 **LIBERO** 基准的离线数据和预处理管线

> **注意**：本仓库的环境基于 [WMPO](https://github.com/WM-PO/WMPO) 和 [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002) 修改而来，数据集也沿用 RynnVLA-002 的数据处理流程。下面的安装文档会完整覆盖从零搭建环境的全部步骤。

当前发布级仓库结构见 [docs/repository_structure.md](docs/repository_structure.md)，脚本入口见 [scripts/README.md](scripts/README.md)，配置注册表见 [configs/README.md](configs/README.md)。

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

当前流水线先收束到 **frozen shared backbone + action-hidden WM**。VLA 和 WM 共享从 observation 到 pi0 action hidden 的整段表征，WM 不再另接一个独立 pixel CNN encoder：

```
┌──────────────────────── Stage 0: 数据准备 ────────────────────────┐
│ LIBERO HDF5 -> no-op 过滤 / 图像抽取 / task language / state/action │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────── Stage 1: pi0 VLA action head SFT ────────────────┐
│ Config:   vla_pi0_query.yaml                                     │
│ Script:   scripts/train_vla.sh                                   │
│ 输出:     frozen VLA ckpt, action_head_type=pi0_query             │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────── Stage 2: 预计算 action-hidden sidecar ───────────────┐
│ Script: scripts/preprocess/preprocess_rynn_pixel_hidden.py       │
│ obs + language + state                                           │
│   -> shared VLA backbone + pi0 action-query block                │
│   -> action_hidden [H, 1024]                                     │
│   -> flatten to obs_embedding [H*1024]                           │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────── Stage 3: frozen action-hidden DreamerV3 WM ──────────┐
│ Config:    world_model_dinowm_chunk.yaml                         │
│ Script:    scripts/train_wm.sh                                   │
│ Runner: RynnDinoWMRunner                                   │
│ Model:     ChunkAwareRynnDinoWMWorldModel                        │
│ 学习:      chunk transition, hidden reconstruction, reward        │
└───────────────────────────────────────────────────────────────────┘
```

### 关键注意点

**1. action hidden 有两个身份**

对 VLA，它是 action regressor 的输入；对 WM，它是 DreamerV3 RSSM posterior encoder 的 observation embedding。当前第一版把 action hidden 预计算成 sidecar，训练 WM 时固定 VLA backbone。

**2. 当前实现边界：先 frozen，再 actor**

joint finetune 需要让 WM loss 和 VLA action loss 同时作用到共享 backbone，风险更高，容易把已可用的 VLA action 表达拉坏。本仓库当前主线先跑通 frozen action-hidden WM，再用 action-hidden actor 训练下游策略。

**3. DreamerVLA actor-critic 当前走 action-hidden head**

当前 actor 配置是 `configs/dreamervla_rynn_dino_wm_wmpo_outcome.yaml`，脚本入口是 `scripts/train_dreamervla.sh`。

**4. pixel WM / token WM / LaDiWM 是 secondary**

这些路线保留用于 baseline / reproducibility；旧 pooled hidden、TransDreamer/TSSM token WM、semantic bottleneck 和旧 scalar actor 分支已从训练入口删除。路线列表见 `docs/wm_training_routes.md`。

---

## 项目结构

发布时推荐按下面的边界理解仓库：

```text
DreamerVLA/
├── dreamer_vla/    # Python 源码包；runner/model/dataset/algorithm 都在这里
├── configs/        # Hydra 实验配置；当前主线和历史 ablation 见 configs/README.md
├── scripts/        # 训练、评估、预处理、诊断入口；稳定入口见 scripts/README.md
├── tests/          # unit_tests / e2e_tests
├── docs/           # 架构说明、实验结论、发布结构说明
├── data/           # 运行时数据、权重、输出；被 gitignore 忽略
└── third_party/    # LIBERO / OpenVLA-OFT / robosuite 等本地依赖；被 gitignore 忽略
```

当前 LIBERO-goal / pi0 action-hidden 主线要求 VLA base、pi0 VLA action head、action-hidden sidecar、`action_head_type=pi0_query` 和 `time_horizon` 完全一致；具体路径和约束见 `configs/README.md`。

```text
DreamerVLA/
├── configs/                        # 实验配置 (Hydra YAML)
│   ├── vla_pi0_query.yaml
│   ├── world_model_dinowm_chunk.yaml
│   ├── dreamervla_rynn_dino_wm_wmpo_outcome.yaml
│   ├── eval_libero_vla.yaml
│   └── ...
├── data/                           # 运行时数据（不入 git）
│   ├── ckpts/                      # 模型权重
│   ├── dataset/                    # LIBERO / CALVIN / 预处理数据 / metainfo
│   └── outputs/                    # 训练输出（checkpoint, log）
├── docs/                           # 技术文档
├── third_party/                    # 本地第三方 checkout / wheel
│   ├── LIBERO/                      # LIBERO 基准本地 checkout
│   ├── openvla-oft/                # 官方 OpenVLA-OFT 评估代码
│   └── openvla-oft-lightweight/    # DreamerVLA 内部轻量兼容导入树
├── scripts/                        # 训练 / 预处理 / 评估入口脚本（薄 wrapper）
│   ├── eval_libero_vla.sh          # VLA / Dreamer checkpoint LIBERO 评估
│   ├── train_wm.sh                 # 统一 World Model 训练入口
│   ├── train_dreamervla.sh         # Dreamer-VLA 训练
│   └── preprocess/                 # 各步骤预处理脚本
├── dreamer_vla/                            # 源代码
│   ├── algorithms/                 # Dreamer-VLA, PPO/GRPO
│   ├── cli/                        # scripts/ 对应的命令行实现
│   ├── dataset/                    # 数据集加载器和在线 rollout dumper
│   ├── envs/                       # LIBERO 环境封装
│   ├── models/                     # 模型定义
│   │   ├── actor/                  # Actor / policy 网络
│   │   ├── chameleon_model/        # Chameleon 视觉语言模型
│   │   ├── encoder/                # RynnVLA 编码器封装
│   │   ├── world_model/            # DreamerV3 / Chameleon WM
│   │   └── critic/                 # Critic 网络
│   ├── preprocess/                 # 数据预处理逻辑和 xllmx 预处理适配
│   ├── trainer/                    # 分布式训练器
│   ├── utils/                      # 工具函数
│   └── runners/                    # 实验 Runner
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
cd /mnt/data/spoil/workspace/DreamerVLA
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
# 如果 WMPO 仓库中有 third_party/egl_probe 目录，则从那里安装
cd /home/user01/yuxinglei/workspace/WMPO/third_party/egl_probe

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

# TensorNVMe（如果 WMPO 仓库中有 third_party/TensorNVMe）
cd /home/user01/yuxinglei/workspace/WMPO/third_party/TensorNVMe
pip install -e .

# NVIDIA APEX（从源码编译）
cd /home/user01/yuxinglei/workspace/WMPO/third_party
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-build-isolation .
```

### 8. 安装 LIBERO

本仓库自带一个 LIBERO 的本地 checkout（位于 `third_party/LIBERO/` 目录），已修复了上游的 editable-install 问题（[Issue #31](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/31)）。

#### 安装

```bash
cd /mnt/data/spoil/workspace/DreamerVLA/third_party/LIBERO
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
cd /mnt/data/spoil/workspace/DreamerVLA

# 基础环境
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# Python 依赖
python -c "import h5py, hydra, omegaconf, transformers; print('python deps ok')"

# 编码器 import
python -c "from dreamer_vla.models.encoder.rynnvla_encoder import RynnVLAEncoder; print('encoder import ok')"

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
CKPT_DIR=/mnt/data/spoil/workspace/DreamerVLA/data/ckpts

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
CKPT_DIR=/mnt/data/spoil/workspace/DreamerVLA/data/ckpts

# VLA 模型权重（256 分辨率，当前主线使用 libero_goal）
hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${CKPT_DIR}/VLA_model_256/libero_goal" \
  --include "VLA_model_256/libero_goal/*"

# Action World Model 权重（512 分辨率，可选；当前 action-hidden 主线不直接依赖）
hf download Alibaba-DAMO-Academy/RynnVLA-002 \
  --repo-type model \
  --local-dir "${CKPT_DIR}/Action_World_model_512/libero_goal" \
  --include "Action_World_model_512/libero_goal/*"
```

如需批量下载，建议把上面的 `hf download` 命令整理成你自己的本地脚本；当前仓库不再保留活动的下载 shell wrapper。

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
│   └── libero_goal/
└── Action_World_model_512/
    └── libero_goal/
```

快速检查：

```bash
ls data/ckpts/chameleon/tokenizer/
ls data/ckpts/VLA_model_256/libero_goal/
ls data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/
```

如果权重下载到了其他位置，可以创建软链接或修改 `configs/` 下 YAML 中的路径。

---

## 数据集下载与预处理

### 1. 下载 LIBERO 数据集

LIBERO 数据集为 HDF5 格式。推荐使用 Hugging Face 源下载：

```bash
cd /mnt/data/spoil/workspace/DreamerVLA

# 下载全部数据集（libero_goal, libero_spatial, libero_object, libero_100）
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface

# 或只下载特定子集
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_goal --use-huggingface
```

默认下载到 `third_party/LIBERO/libero/datasets/` 目录下。也可以通过 `--download-dir` 指定其他路径。

> **提示**：如果已经在 RynnVLA-002 仓库中下载过 LIBERO 数据集，可以直接创建软链接而无需重复下载：
> ```bash
> ln -s /home/user01/yuxinglei/workspace/RynnVLA-002/third_party/LIBERO/libero/datasets \
>       /mnt/data/spoil/workspace/DreamerVLA/data/libero/datasets
> ```

### 2. 数据预处理管线

当前活动预处理入口放在 `scripts/preprocess/`，预处理实现放在
`dreamer_vla/preprocess/`。常用入口包括：

```bash
python scripts/preprocess/preprocess_rynn_pixel_hidden.py --help
python scripts/preprocess/preprocess_oft_action_hidden.py --help
python scripts/preprocess/preprocess_remaining_steps_reward.py --help
python scripts/preprocess/build_classifier_shards_from_demos.py --help
```

历史五步 shell 管线已经移到 `scripts/archive/uncertain_shells/`。不要把归档脚本作为新实验默认入口；新的训练 route 应直接复用
`configs/task/*.yaml` 中的数据路径和 sidecar 约束。

---

## 路径配置

本仓库在配置文件中使用了绝对路径，在新机器上**必须手动修改**。需要检查和更新的文件：

| 文件 | 需更新的字段 |
|------|-------------|
| `configs/vla_pi0_query.yaml` | `training.out_dir`, `init.vla_ckpt_path`, `encoder.*_path`, `dataset.*` |
| `configs/world_model_dinowm_chunk.yaml` | `init.*`, `dataset.hidden_dir`, `dataset.expected_*`, `world_model.*` |
| `configs/dreamervla_rynn_dino_wm_wmpo_outcome.yaml` | `init.*`, `dataset.hidden_dir`, `policy.time_horizon`, `algorithm.*` |

通用规则：将所有 `/mnt/data/spoil/workspace/DreamerVLA` 替换为你的实际项目根目录即可。

---

## Preencode vs Pretokenize 说明

这是两个**完全不同**的数据流程，不要混用：

| | Preencode | Pretokenize |
|---|-----------|-------------|
| **含义** | 预先跑 VLA 编码器，保存连续特征 | 预先保存离散 token 序列 |
| **输出** | `obs_embedding`, `action`, `reward` 等连续张量 | `input_ids`, `labels` 等 token ID |
| **用途** | World Model 训练 | VLA token-level SFT 训练 |
| **数据集** | `preencode_sft_dataset.py` | `pretokenize_dataset.py` |
| **Runner** | 旧 preencode 分支已移除 | `pretokenize_vla_runner.py` / `VLASFTRunner` |

**判断规则**：batch 里是 `obs_embedding` 等连续特征 → preencode；是 `input_ids/labels` → pretokenize。

---

## 训练

所有训练脚本使用 `torchrun` 进行分布式训练，配合 Hydra 读取 YAML 配置。

### VLA SFT 训练

基于 pretokenize 数据的 VLA 监督微调：

```bash
conda activate dreamervla
cd /mnt/data/spoil/workspace/DreamerVLA

# 当前 pi0 action-query VLA head
CONFIG=vla_pi0_query bash scripts/train_vla.sh

# 自定义 GPU 数量和配置
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG=vla_pi0_query \
  bash scripts/train_vla.sh
```

### World Model 训练

当前主线 World Model 训练走 pi0 action-hidden pipeline。默认先打印命令，不启动长任务：

```bash
# 1. 预计算 pi0 action-hidden sidecar
python scripts/preprocess/preprocess_rynn_pixel_hidden.py --help

# 2. 用 frozen action hidden 训练 chunk-aware DINO-WM
CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh
```

也可以直接调用 WM wrapper：

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh
```

`scripts/train_wm.sh` 是统一 WM wrapper。当前活动入口见
`configs/README.md`。

### Dreamer-VLA 完整训练

Action-hidden route 的 DreamerVLA actor-critic stage 使用 pi0 action-hidden head：

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh
```

如果只想调用通用 wrapper：

```bash
CONFIG=dreamervla_rynn_dino_wm_actor_critic \
  bash scripts/train_dreamervla.sh
```

---

## 评估

在 LIBERO 环境中对训练好的 VLA checkpoint 进行 rollout 评估：

```bash
conda activate dreamervla
cd /mnt/data/spoil/workspace/DreamerVLA

bash scripts/eval_libero_vla.sh \
    eval.ckpt_path=data/outputs/vla/checkpoints/example.ckpt \
    eval.task_suite_name=libero_goal \
    eval.num_episodes_per_task=10 \
    training.device=cuda:0
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
