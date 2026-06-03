# DreamerVLA Setup

本文给出从空环境到 LIBERO 评估的最短复现路径。所有命令默认在单机多 GPU Linux 上运行，并默认项目路径为：

```bash
export DVLA_ROOT=/mnt/data/spoil/workspace/DreamerVLA
cd "$DVLA_ROOT"
```

如果仓库不在这个路径，先把 `configs/` 和稳定脚本中的旧绝对路径替换成你的路径：

```bash
OLD=/mnt/data/spoil/workspace/DreamerVLA
NEW=/abs/path/to/DreamerVLA
rg -l "$OLD" configs scripts | xargs sed -i "s#${OLD}#${NEW}#g"
```

## 1. 环境

推荐版本：

- Ubuntu 20.04+，NVIDIA GPU，CUDA 12.x
- Python 3.11
- PyTorch 2.5.1 + CUDA 12.4
- `numpy==1.26.4`
- `transformers==4.40.1`

```bash
conda create -n dreamervla python=3.11 -y
conda activate dreamervla

pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -e "$DVLA_ROOT"
pip install -r "$DVLA_ROOT/requirements.txt"
pip install transformers==4.40.1
```

仿真和训练常用环境变量：

```bash
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

系统 OpenGL 依赖：

```bash
sudo apt update
sudo apt install -y libgl1 libopengl0 libgl1-mesa-dri libgl1-mesa-glx \
  libosmesa6-dev libosmesa6 ffmpeg
```

安装本地第三方包：

```bash
# LIBERO
cd "$DVLA_ROOT/third_party/LIBERO"
python -m pip install --no-build-isolation -e .

# egl_probe，若 CMake 版本报错，先放宽 cmake_minimum_required
cd "$DVLA_ROOT/third_party/egl_probe"
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
  egl_probe/CMakeLists.txt || true
python -m pip install --no-build-isolation .

# 可选：部分历史/WMPO 路线会用到
pip install -e "$DVLA_ROOT/third_party/TensorNVMe" || true
pip install -v --no-build-isolation "$DVLA_ROOT/third_party/apex" || true
```

`flash-attn` 可选。若需要，用和 `torch==2.5.1/cu12/cp311` 匹配的 wheel，例如：

```bash
python -c "import torch; print(torch.__version__, torch.compiled_with_cxx11_abi())"
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.1.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

环境验证：

```bash
cd "$DVLA_ROOT"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import h5py, hydra, omegaconf, transformers; print('deps ok')"
python -c "import libero; print('libero ok')"
python -c "from dreamer_vla.models.encoder.rynnvla_encoder import RynnVLAEncoder; print('encoder ok')"
```

## 2. 权重

先登录 Hugging Face：

```bash
hf auth login
```

把权重下载到 `data/ckpts` 根目录，最终路径要匹配 `configs/task/*.yaml`：

```bash
mkdir -p "$DVLA_ROOT/data/ckpts"

# Chameleon tokenizer / base model
hf download Alibaba-DAMO-Academy/WorldVLA --repo-type model \
  --local-dir "$DVLA_ROOT/data/ckpts" \
  --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*"

# Lumina tokenizer
hf download Alpha-VLLM/Lumina-mGPT-7B-768 --repo-type model \
  --local-dir "$DVLA_ROOT/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"

# RynnVLA-002 VLA ckpt；按需把 libero_goal 换成 libero_object/libero_spatial/libero_10
hf download Alibaba-DAMO-Academy/RynnVLA-002 --repo-type model \
  --local-dir "$DVLA_ROOT/data/ckpts" \
  --include "VLA_model_256/libero_goal/*" "Action_World_model_512/libero_goal/*"
```

检查关键文件：

```bash
test -f "$DVLA_ROOT/data/ckpts/chameleon/tokenizer/text_tokenizer.json"
test -f "$DVLA_ROOT/data/ckpts/chameleon/tokenizer/vqgan.yaml"
test -d "$DVLA_ROOT/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
test -d "$DVLA_ROOT/data/ckpts/VLA_model_256/libero_goal"
```

## 3. LIBERO 数据

下载原始 LIBERO HDF5：

```bash
cd "$DVLA_ROOT"
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_goal --use-huggingface
```

确认 `~/.libero/config.yaml` 的 `datasets:` 指向实际 LIBERO 数据目录，例如：

```bash
grep '^datasets:' ~/.libero/config.yaml
```

## 4. 数据处理

下面以 `libero_goal` 为例。其他 suite 把 `SUITE` 改成 `libero_object`、`libero_spatial` 或 `libero_10`。

```bash
export SUITE=libero_goal
export RAW_LIBERO="$DVLA_ROOT/third_party/LIBERO/libero/datasets/$SUITE"
export HDF5_DIR="$DVLA_ROOT/data/processed_data/${SUITE}_no_noops_t_256"
export REWARD_DIR="$DVLA_ROOT/data/processed_data/${SUITE}_no_noops_t_256_pi06_remaining_reward"
export HIDDEN_DIR="$DVLA_ROOT/data/processed_data/${SUITE}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
export META_JSON="$DVLA_ROOT/data/processed_data/${SUITE}_metainfo.json"
```

### 4.1 生成 no-op 过滤后的 HDF5

这一步会 replay LIBERO demos，过滤 no-op 和失败 demo，并写出 256 分辨率 HDF5：

```bash
cd "$DVLA_ROOT/dreamer_vla/preprocess/libero_utils"
python regenerate_libero_dataset_filter_no_op.py \
  --libero_task_suite "$SUITE" \
  --libero_raw_data_dir "$RAW_LIBERO" \
  --libero_target_dir "$HDF5_DIR" \
  --image_resolution 256

mkdir -p "$DVLA_ROOT/data/processed_data"
mv "${SUITE}_metainfo.json" "$META_JSON"
```

### 4.2 生成 VLA SFT 所需 pretokenize 数据

该脚本会生成 image/state/action 目录、conversation JSON、token pkl、manifest 和 `data/configs/<suite>/*.yaml`。

```bash
cd "$DVLA_ROOT"
SUITES="$SUITE" GPUS=0,1 PRETOKENIZE_PROCS=8 FORCE=0 \
  bash scripts/preprocess/process_all_libero_data.sh
```

如果你的 conda 不在 `/home/user01/miniconda3`，先修改 `scripts/preprocess/process_all_libero_data.sh` 顶部的 `source .../conda.sh` 和 `PYTHON=...` 两行。

产物：

```text
data/processed_data/convs/
data/processed_data/tokens/
data/processed_data/concate_tokens/
data/configs/<suite>/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml
```

校验：

```bash
python -m dreamer_vla.preprocess.validate_pretokenized \
  --tokens-dir "$DVLA_ROOT/data/processed_data/tokens" \
  --sample-every 200
```

### 4.3 生成 remaining-steps reward HDF5

WM 和 DreamerVLA 训练读取 `task.hdf5_reward_dir`：

```bash
cd "$DVLA_ROOT"
python scripts/preprocess/preprocess_remaining_steps_reward.py \
  --input-dir "$HDF5_DIR" \
  --output-dir "$REWARD_DIR" \
  --metainfo-json "$META_JSON" \
  --overwrite
```

### 4.4 生成 pi0 legacy action-hidden sidecar

WM 和 DreamerVLA 训练读取 `task.pi0_legacy_action_hidden_dir`。为避免脚本默认路径和机器路径不一致，显式传入所有路径：

```bash
cd "$DVLA_ROOT"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/preprocess/preprocess_rynn_pixel_hidden.py \
  --hdf5-dir "$HDF5_DIR" \
  --out-dir "$HIDDEN_DIR" \
  --model-path "$DVLA_ROOT/data/ckpts/VLA_model_256/$SUITE" \
  --tokenizer-path "$DVLA_ROOT/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768" \
  --text-tokenizer-path "$DVLA_ROOT/data/ckpts/chameleon/tokenizer/text_tokenizer.json" \
  --chameleon-vqgan-config "$DVLA_ROOT/data/ckpts/chameleon/tokenizer/vqgan.yaml" \
  --chameleon-vqgan-ckpt "$DVLA_ROOT/data/ckpts/chameleon/tokenizer/vqgan.ckpt" \
  --action-head-type legacy \
  --obs-hidden-source action_query \
  --history 2 \
  --include-state \
  --rotate-images-180 \
  --save-action-hidden \
  --action-dim 7 \
  --time-horizon 5 \
  --overwrite
```

校验 sidecar：

```bash
python - <<'PY'
import os
from pathlib import Path
import h5py
root = Path(os.environ["HIDDEN_DIR"])
path = next(root.glob("*.hdf5"))
with h5py.File(path, "r") as f:
    demo = next(iter(f["data"].values()))
    print(path.name)
    print("complete:", bool(f.attrs.get("complete", False)))
    print("obs_embedding:", demo["obs_embedding"].shape)
    print("action_hidden_states:", demo["action_hidden_states"].shape)
PY
```

期望 `obs_embedding` 最后一维为 `35840`，`action_hidden_states` 为 `[T, 35, 1024]`。

### 4.5 更新 task config

确认 `configs/task/${SUITE}.yaml` 中这些字段指向刚生成的目录：

```yaml
vla_ckpt_path: /mnt/data/spoil/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_goal
pretokenize_config_path: /mnt/data/spoil/workspace/DreamerVLA/data/configs/libero_goal/his_1_third_view_wrist_w_state_1_256_pretokenize.yaml
pretokenize_val_ind_config_path: /mnt/data/spoil/workspace/DreamerVLA/data/configs/libero_goal/his_1_third_view_wrist_w_state_1_256_pretokenize_val_ind.yaml
pretokenize_val_ood_config_path: /mnt/data/spoil/workspace/DreamerVLA/data/configs/libero_goal/his_1_third_view_wrist_w_state_1_256_pretokenize_val_ood.yaml
hdf5_dir: /mnt/data/spoil/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256
hdf5_reward_dir: /mnt/data/spoil/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward
pi0_legacy_action_hidden_dir: /mnt/data/spoil/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
```

## 5. 训练

所有训练都走 Hydra config。常用环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
```

### 5.1 VLA SFT

```bash
cd "$DVLA_ROOT"
OUT_DIR="$DVLA_ROOT/data/outputs/vla/pi0_query/libero_goal_run1" \
CONFIG=vla_pi0_query \
bash scripts/train_vla.sh task=libero_goal
```

快速 smoke：

```bash
OUT_DIR=/tmp/dvla_vla_smoke CONFIG=vla_pi0_query \
bash scripts/train_vla.sh task=libero_goal training.max_train_steps=1 dataloader.num_workers=0
```

### 5.2 World Model

```bash
OUT_DIR="$DVLA_ROOT/data/outputs/worldmodel/dinowm_chunk/libero_goal_run1" \
CONFIG=world_model_dinowm_chunk \
bash scripts/train_wm.sh task=libero_goal training.max_steps=20000
```

最终常用 ckpt：

```text
data/outputs/worldmodel/dinowm_chunk/<run>/ckpt/latest.ckpt
data/outputs/worldmodel/dinowm_chunk/<run>/ckpt/step_00020000.ckpt
```

快速 smoke：

```bash
OUT_DIR=/tmp/dvla_wm_smoke CONFIG=world_model_dinowm_chunk \
bash scripts/train_wm.sh task=libero_goal training.max_steps=1 dataloader.num_workers=0
```

### 5.3 LatentSuccessClassifier

`dreamervla_rynn_dino_wm_wmpo_outcome` 需要 `init.classifier_state_ckpt`。先训练 classifier：

```bash
OUT_DIR="$DVLA_ROOT/data/outputs/dreamervla/outcome_classifier/libero_goal/run1" \
CONFIG=latent_classifier_libero_goal_chunk \
bash scripts/train_wm.sh
```

如果还没有 failure demo 目录，可以先让配置只使用成功 demo：

```bash
OUT_DIR="$DVLA_ROOT/data/outputs/dreamervla/outcome_classifier/libero_goal/run1" \
CONFIG=latent_classifier_libero_goal_chunk \
bash scripts/train_wm.sh \
  data.failure_dir_raw=null \
  data.failure_dir_hidden=null
```

产物在：

```text
data/outputs/dreamervla/outcome_classifier/libero_goal/<run>/ckpt/best_*.ckpt
data/outputs/dreamervla/outcome_classifier/libero_goal/<run>/ckpt/latest.ckpt
```

### 5.4 DreamerVLA

使用上一步的 WM ckpt 和 classifier ckpt：

```bash
export WM_CKPT="$DVLA_ROOT/data/outputs/worldmodel/dinowm_chunk/libero_goal_run1/ckpt/latest.ckpt"
export CLS_CKPT="$DVLA_ROOT/data/outputs/dreamervla/outcome_classifier/libero_goal/run1/ckpt/latest.ckpt"

OUT_DIR="$DVLA_ROOT/data/outputs/dreamervla/wmpo_outcome/libero_goal_run1" \
CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome \
bash scripts/train_dreamervla.sh \
  task=libero_goal \
  init.world_model_state_ckpt="$WM_CKPT" \
  init.classifier_state_ckpt="$CLS_CKPT"
```

若只想避开 outcome classifier 依赖，可先跑 actor-critic route：

```bash
OUT_DIR="$DVLA_ROOT/data/outputs/dreamervla/actor_critic/libero_goal_run1" \
CONFIG=dreamervla_rynn_dino_wm_actor_critic \
bash scripts/train_dreamervla.sh \
  task=libero_goal \
  init.world_model_state_ckpt="$WM_CKPT"
```

## 6. 评估

### 6.1 VLA checkpoint

```bash
export VLA_CKPT="$DVLA_ROOT/data/outputs/vla/pi0_query/libero_goal_run1/ckpt/latest.ckpt"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  init.vla_ckpt_path="$DVLA_ROOT/data/ckpts/VLA_model_256/libero_goal" \
  eval.ckpt_path="$VLA_CKPT" \
  eval.ckpt_kind=vla \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  eval.action_steps=5 \
  training.device=cuda:0
```

### 6.2 Dreamer checkpoint

```bash
export DREAMER_CKPT="$DVLA_ROOT/data/outputs/dreamervla/wmpo_outcome/libero_goal_run1/ckpt/latest.ckpt"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  init.vla_ckpt_path="$DVLA_ROOT/data/ckpts/VLA_model_256/libero_goal" \
  eval.ckpt_path="$DREAMER_CKPT" \
  eval.ckpt_kind=dreamer \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  eval.action_steps=5 \
  training.device=cuda:0
```

输出默认在：

```text
data/outputs/eval/eval_libero_vla/
```

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: libero` | 重新执行 `python -m pip install --no-build-isolation -e third_party/LIBERO`，并在 `/tmp` 下验证 import |
| LIBERO 找不到数据 | 检查 `~/.libero/config.yaml` 的 `datasets:` |
| CUDA 不可用 | 检查驱动、`nvidia-smi`、PyTorch wheel 是否为 cu124 |
| `xformers` 冲突 | 确认 `torch==2.5.1`，再安装 `requirements.txt` |
| `flash-attn` 编译失败 | 使用预编译 wheel，或暂时跳过 |
| 训练报路径不存在 | 检查 `configs/task/*.yaml` 和 `init.*_ckpt` 是否仍是旧机器绝对路径 |
| WM 读取 sidecar 报 schema mismatch | 确认 hidden sidecar 使用 `--action-head-type legacy --history 2 --include-state --rotate-images-180` |
| DDP 卡住 | 先用 `NGPU=1` 和 `training.max_steps=1` smoke；再检查 rank0 日志和 batch/NaN |

## 8. 最小成功标准

复现到可评估状态至少应满足：

```bash
pytest tests/unit_tests -q
test -d data/ckpts/VLA_model_256/libero_goal
test -d data/processed_data/libero_goal_no_noops_t_256
test -d data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward
test -d data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
bash scripts/train_wm.sh task=libero_goal training.max_steps=1 dataloader.num_workers=0
```
