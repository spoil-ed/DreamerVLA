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

## Pipeline Entry

完整 cold-start pipeline 入口：

```bash
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray task=goal
```

常用 shell 包装：

```bash
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray task=goal
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=noray task=goal
```

## Main Runner Targets

当前仓库导出的关键 Runner 包括：

- `CollectRolloutsRunner`：no-Ray collection。
- `ColdStartRayCollectRunner`：Ray collection。
- `OnlineCotrainPipelineRunner`：sync warmup + online cotrain pipeline。
- `ManualCotrainRayRunner`：当前 manual Ray cotrain route。
- `OnlineCotrainRunner` / `OnlineCotrainRayRunner`：同步与可选 Ray online cotrain 路线。
- `LatentClassifierRunner`：classifier 单组件训练。
- `EmbodiedEvalRunner`：VLA / DreamerVLA 统一评估入口。

完整 replay 的 world-model 单组件训练复用
`OnlineCotrainPipelineRunner` 的 warmup 路径，不另设第二套 WM Runner 接口。

## Run Artifacts

每次运行应落在一个 run root 下。Runner 产物通常包括：

- `resolved_config.yaml`
- `run_manifest.json`
- `checkpoints/`
- `log/`
- `video/`
- `diagnostics/`

pipeline 额外使用 collection root、warmup checkpoint 和 cotrain sub-root。不要把核心产物散落到未声明路径。
