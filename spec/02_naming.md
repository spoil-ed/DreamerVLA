# Component Map And Names

名称按 runtime 角色命名，不按实验历史命名。同一个角色只保留一个正式名称。

## Runtime Roles

| 名称 | 当前含义 |
| --- | --- |
| `Runner` | 一个 train/eval job 的生命周期所有者。 |
| `EnvGroup` | 环境交互 group；实际可拆成 `RealEnvGroup` 和 `WMEnvGroup`。 |
| `RealEnvWorker` | 真实 LIBERO 环境 step。 |
| `WMEnvWorker` | latent world-model environment step。 |
| `RolloutGroup` | 行为策略推理副本，no-grad/eval。 |
| `RolloutWorker` / `MultiStepRolloutWorker` | 接 obs，生成 action chunk、old logprob、forward inputs。 |
| `ActorGroup` | VLA policy training，负责 PPO、backward、optimizer、FSDP。 |
| `EmbodiedFSDPActor` | ActorGroup 内的 FSDP actor worker。 |
| `LearnerGroup` | world model 与 classifier/reward model training。 |
| `LearnerWorker` | LearnerGroup 内部 worker。 |
| `ReplayGroup` / `ReplayWorker` | 可选 replay service，用于数据、warmup、resume、bootstrap。 |

## Package Roles

| 包 | 内容 |
| --- | --- |
| `dreamervla.algorithms` | PPO/LUMOS 类更新、registry、reward/verifier 协议。 |
| `dreamervla.dataset` | online replay、rollout dump、HDF5 dataset、manifest。 |
| `dreamervla.envs` | LIBERO train/eval env 和 world-model env。 |
| `dreamervla.models.actor` | VLA actor 与 latent-to-action actor。 |
| `dreamervla.models.encoder` | VLA/OFT/RynnVLA encoder。 |
| `dreamervla.models.world_model` | Dreamer/TSSM/DINO world model 实现。 |
| `dreamervla.models.reward` | latent success classifier。 |
| `dreamervla.workers.actor` | `EmbodiedFSDPActor` 与 `LearnerWorker`。 |
| `dreamervla.workers.rollout` / `workers.inference` | rollout/inference worker。 |
| `dreamervla.workers.env` | trajectory env worker。 |
| `dreamervla.workers.replay` | replay worker。 |
| `dreamervla.scheduler` | worker group、placement、channel、cluster 抽象。 |
| `dreamervla.hybrid_engines` | FSDP manager 与 weight syncer。 |

## Naming Rules

- 用角色名：`ActorGroup`、`RolloutGroup`、`LearnerGroup`、`EnvGroup`。
- 避免 `new`、`v2`、`manager`、`async` 这类不表达职责的核心名。
- config、metrics、checkpoint 和 tests 中出现的名称视为公共接口。
- 历史路线可保留，但不要让历史名成为新主线的正式概念。
