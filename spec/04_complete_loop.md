# Complete Loop

主线是 OpenVLA-OFT one-trajectory cold start：

```text
collect -> warmup -> online cotrain -> eval
```

## 1. Collect

collection 在真实 LIBERO 中运行当前 VLA/OFT policy，写出 reward shard、hidden sidecar 和
`collection_manifest.json`。

当前入口：

- no-Ray：`experiment=collect_rollouts_onetraj` -> `CollectRolloutsRunner`
- Ray：`experiment=collect_rollouts_ray` -> `ColdStartRayCollectRunner`

collection 只负责产生初始真实数据，不负责训练 world model 或 actor。

## 2. Warmup

cotrain 前先用 collected reward + hidden HDF5 shard 训练：

- world model
- classifier/reward model

sync pipeline 使用 `OnlineCotrainPipelineRunner`。async pipeline 先跑 sync warmup，再把 warmup checkpoint
合并成 manual Ray cotrain 的 init checkpoint。

## 3. Online Cotrain

online cotrain 同时维护三类模型状态：

- Actor/VLA policy：由 ActorGroup 更新。
- Rollout policy replica：由 RolloutGroup 做 no-grad inference，定期从 Actor 同步。
- World model + classifier：由 LearnerGroup 更新，定期同步给 WMEnvWorker。

Actor PPO 的训练数据来自 rollout 组装出的 trajectory，不应隐式从 replay 替代。

## 4. Eval

最终质量以真实 LIBERO eval 为准。WMEnv reward 可以提供训练信号，但不能替代真实环境 success rate。

## Data Flow

```text
Real LIBERO collection
  -> reward/hidden shards + manifest
  -> OnlineReplay / warmup datasets
  -> WM + classifier warmup checkpoints
  -> online cotrain
  -> actor/wm/classifier checkpoints + metrics
  -> real LIBERO eval
```

## Main Configs

- Launcher config：`configs/scripts/coldstart_warmup_cotrain.yaml`
- Sync cotrain experiment：`openvla_onetraj_libero_cotrain_noray`
- Manual Ray experiment：`openvla_onetraj_libero_cotrain_ray`
- Eval config：`configs/scripts/eval_libero_vla.yaml`

## TODO

- 编排解耦：各 stage 的 runner 已经进程级解耦（每个 stage = 一次 `python -m dreamervla.train
  experiment=<x>` -> 一个 `_target_`，无进程内耦合）。对固定、无跨阶段穿线的 recipe，可考虑用薄 shell
  `cmd1 && cmd2 && cmd3` 顺序激活各 runner（更直观）。但跨阶段胶水（路径穿线、`ngpu>1` 的 torchrun 包裹、
  ray/noray collect 选择、episode 级续采、async warmup/online 拆相、manifest）仍应留在可单测的 Python
  launcher，遵循「薄 shell、逻辑走 `python -m`」。落地范围 = 无胶水路径的 shell 便捷封装，不是替换 launcher。
