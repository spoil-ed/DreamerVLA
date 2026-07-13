# Architecture Overview

DreamerVLA 是一个单机 LIBERO VLA + world model 训练栈。Hydra 选择实验配置，
`dreamervla.train` 加载 `_target_` Runner，Runner 拥有一次 train/eval job 的完整生命周期。

主线流程：

```text
collect rollouts -> seed replay/data -> warm up world model + classifier -> online cotrain -> eval
```

## Repository Shape

| 路径 | 职责 |
| --- | --- |
| `dreamervla/train.py` | 统一 Hydra 训练入口：resolve config、validate、实例化 Runner。 |
| `dreamervla/config.py` | 早期配置关系校验。 |
| `dreamervla/launchers/` | 组合多阶段命令；主线是 `coldstart_warmup_cotrain.py`。 |
| `dreamervla/runners/` | 训练/评估 job 入口；Runner 负责 setup、execute、teardown。 |
| `dreamervla/models/` | VLA、encoder、actor、critic、world model、reward/classifier。 |
| `dreamervla/dataset/`, `dreamervla/preprocess/` | rollout、sidecar、HDF5、manifest 和预处理。 |
| `dreamervla/envs/` | LIBERO 真实环境和 world-model environment 包装。 |
| `dreamervla/workers/`, `scheduler/`, `hybrid_engines/` | Ray/manual cotrain 的 worker、placement、channel、FSDP 和权重同步。 |
| `configs/` | Hydra source of truth。 |
| `scripts/` | 薄 shell 入口，转发到 Python/Hydra。 |
| `tests/` | 单元测试和 e2e/smoke 覆盖。 |

## Current Mainline

当前主线围绕 OpenVLA-OFT one-trajectory cold start：

```text
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=noray
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray
```

launcher 根据
`configs/scripts/coldstart_warmup_cotrain.yaml` 生成 collection 和 cotrain 阶段命令。

## Stable Boundaries

- Hydra 决定组件和参数；训练 loop 不硬编码具体模型类。
- Runner 拥有一个 job；Worker 只做自己的 runtime 角色。
- ActorGroup 训练 VLA；RolloutGroup 做 no-grad policy inference。
- LearnerGroup 训练 world model 和 classifier/reward model。
- EnvGroup 只负责真实环境或 WMEnv step 和 trajectory assembly。
- Replay 是数据与 resume 设施，不是 Actor PPO 的隐藏替代通道。
