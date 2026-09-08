# Component Map And Names

名称按 runtime 角色命名，不按实验时间线命名。同一个角色只保留一个正式名称。

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
| `ReplayGroup` / `ReplayWorker` | 可选临时 replay service，用于数据、warmup 和 bootstrap；不参与 cotrain resume。 |

## Package Roles

| 包 | 内容 |
| --- | --- |
| `dreamervla.algorithms` | PPO/LUMOS 类更新、registry、reward/verifier 协议。 |
| `dreamervla.dataset` | `base/` 数据读取、LIBERO 适配、classifier dataset、`storage/` dump/manifest。 |
| `dreamervla.envs` | LIBERO train/eval env 和 world-model env。 |
| `dreamervla.algorithms.actor` | VLA actor 与 latent-to-action actor。 |
| `dreamervla.models.embodiment` | VLA/OFT/π0.5 encoder 和 policy。 |
| `dreamervla.models.embodiment.world_model` | Dreamer/TSSM/DINO world model 实现。 |
| `dreamervla.algorithms.critic` | critic 和 latent success classifier。 |
| `dreamervla.workers.actor` | `EmbodiedFSDPActor` 与 `LearnerWorker`。 |
| `dreamervla.workers.rollout` / `workers.inference` | rollout/inference worker。 |
| `dreamervla.workers.env` | trajectory env worker。 |
| `dreamervla.workers.replay` | replay worker。 |
| `dreamervla.scheduler` | worker group、placement、channel、cluster 抽象。 |
| `dreamervla.hybrid_engines` | FSDP manager 与 weight syncer。 |
| `dreamervla.runtime` | rollout、replay、training、evaluation、envs 和共享工作流支持。 |
| `dreamervla.utils` | checkpoint、config、logging、training、integrations 和 visualization 工具。 |
| `dreamervla.diagnostics` | checks、evaluation、benchmarks 和可供 Ray 导入的 fixtures。 |

## Naming Rules

- 用角色名：`ActorGroup`、`RolloutGroup`、`LearnerGroup`、`EnvGroup`。
- 避免 `new`、`v2`、`manager`、`async` 这类不表达职责的核心名。
- config、metrics、checkpoint 和 tests 中出现的名称视为公共接口。
