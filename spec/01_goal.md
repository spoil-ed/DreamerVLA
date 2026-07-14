# Goal And Entrypoints

DreamerVLA 的目标是把 LIBERO 上的 VLA policy、world model、classifier/reward model 组织成可恢复、
可验证、可扩展的单机训练闭环。

## Train Entry

统一训练入口：

```bash
python -m dreamervla.train experiment=<name> task=<suite>
```

`dreamervla.train` 做四件事：

1. 注册 resolver。
2. 解析并校验 Hydra config。
3. 根据 `cfg._target_` 加载 Runner。
4. 执行 `setup -> execute -> teardown`。

## Mainline Entries

```bash
python -m dreamervla.train \
  experiment=collect_rollouts task=openvla_onetraj_coldstart_libero
bash scripts/experiments/cotrain/train.sh
bash scripts/experiments/cotrain/eval.sh eval.ckpt_path=/path/to/checkpoint
```

## Main Runner Targets

当前仓库导出的关键 Runner 包括：

- `RolloutCollectionRunner`：Ray-backed collection。
- `CotrainRunner`：当前 staged cotrain 主线。
- `WorldModelTrainingRunner`：world model 单组件训练。
- `SuccessClassifierTrainingRunner`：classifier 单组件训练。
- `LIBEROVLAEvaluationRunner`：VLA / DreamerVLA 统一评估入口。

具体 classifier model 和 dataset target 由 `configs/task` 的
`task.classifier` 统一选择；`classifier=dreamer-cls` 不绑定数据集名称。

## Run Artifacts

每次运行应落在一个 run root 下。Runner 产物通常包括：

- `resolved_config.yaml`
- `run_manifest.json`
- `checkpoints/`
- `log/`
- `video/`
- `diagnostics/`

pipeline 额外使用 collection root、warmup checkpoint 和 cotrain sub-root。不要把核心产物散落到未声明路径。
