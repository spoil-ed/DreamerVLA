# OpenVLA One-Trajectory LIBERO Mainline

当前主线只有一套公开路线：

```text
collect -> train WM + classifier -> cotrain -> eval
```

collection 与 cotrain 都以 Ray 为实现后端。Ray 是主线内部实现，不是需要用户选择的
另一套 experiment，因此公开名称没有 `_ray` 后缀，也不再提供同能力的 `_noray` 路线。

## 1. 环境与数据

```bash
export DVLA_ROOT=/path/to/DreamerVLA
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
conda activate dreamervla
cd "${DVLA_ROOT}"
```

安装、下载与预处理仍通过冻结的脚本入口完成：

```bash
bash scripts/install_env.sh
bash scripts/download_assets.sh
bash scripts/preprocess_libero.sh
```

## 2. Collect

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/collect_rollouts/train.sh \
  task=openvla_onetraj_coldstart_libero
```

该入口直接选择 `experiment=collect_rollouts`。
`RolloutCollectionRunner` 使用 Ray worker 采集真实 LIBERO 轨迹，输出 reward shard、
hidden-token sidecar 和 `collection_manifest.json`。资源、episode 数和输出目录都来自
Hydra；shell/命令中只写必要 override。

其他 suite 使用对应 task：

- `openvla_onetraj_coldstart_libero_object`
- `openvla_onetraj_coldstart_libero_spatial`
- `openvla_onetraj_coldstart_libero_10`

## 3. 独立训练 World Model 与 Classifier

World model 在同一入口下切换实现：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/experiments/world_model_training/train.sh --config dreamer-wm

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/experiments/world_model_training/train.sh --config dino-wm
```

成功分类器：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/experiments/classifier_training/train.sh
```

Classifier 的配置边界是：

```text
experiment
  -> classifier=dreamer-cls
  -> task.classifier.model
  -> task.classifier.dataset.{train,validation}
  -> task.classifier.input
```

`configs/classifier/dreamer-cls.yaml` 不绑定数据集或上游 VLA 名称。具体 model target、
dataset target 和 token 输入契约由所选 `configs/task/*.yaml` 统一提供；数据目录和采样
协议由 experiment 的 `data` 段提供。训练 runner 通过 Hydra instantiate 构造模型和数据集。

## 4. Cotrain

将独立训练得到的 WM/CLS checkpoint 显式传给主线：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/world_model.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

该入口选择 `experiment=openvla_libero` 和 public
`dreamervla.runners.CotrainRunner`。内部仍使用 Ray 的 `LearnerGroup`、`ActorGroup`、
`RolloutGroup` 与 `EnvGroup`，但这些是 backend contract，不是公开路线名称。

Hydra override 可以直接追加到命令后，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/world_model.ckpt \
  --cls_ckpt /path/to/classifier.ckpt \
  manual_cotrain.global_steps=100 \
  training.out_dir=/path/to/run
```

## 5. Eval

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/cotrain/checkpoint
```

评估入口选择 `experiment=eval_cotrain`。真实 LIBERO 协议、task 数、每 task episode 数、
并行环境数和渲染后端都在 Hydra 配置中定义。

## Config Ownership

| 配置 | 职责 |
| --- | --- |
| `configs/experiment/collect_rollouts.yaml` | collection 完整 recipe |
| `configs/experiment/dreamer-wm.yaml` / `dino-wm.yaml` | WM recipe 选择 |
| `configs/classifier/dreamer-cls.yaml` | classifier 角色与结构 |
| `configs/task/*.yaml` 的 `task.classifier` | classifier model、dataset、输入契约 |
| `configs/experiment/openvla_libero.yaml` | 完整 cotrain recipe |
| `configs/experiment/eval_cotrain.yaml` | cotrain checkpoint 评估协议 |

参数只在 Hydra 中维护；shell 脚本保持单命令入口，不复制 batch size、learning rate、
horizon、checkpoint cadence 或资源默认值。
