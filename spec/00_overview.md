# Architecture Overview

状态：目标主线架构入口

`spec/` 是 DreamerVLA 当前 architecture 的根目录。主线判断以
[`99_manual_notes.md`](99_manual_notes.md) 的用户第一性指导为准：它定义目标拓扑、抽象边界
和不可破坏约束，是受保护的第一性来源。除非用户明确要求，AI assistant 不得移动、改写或删除
其中的用户手写内容。

| 文件 | 作用 |
| --- | --- |
| [`00_overview.md`](00_overview.md) | 目标主线架构、核心 group、训练形态、关键边界。 |
| [`01_complete_loop.md`](01_complete_loop.md) | 端到端 loop：collect、warmup、cotrain、eval。 |
| [`02_ray.md`](02_ray.md) | Ray worker graph、placement、channel、同步的实现形态。 |
| [`03_current_implementation.md`](03_current_implementation.md) | 当前代码已落地的 route、group、placement、global step 和验证状态。 |
| [`04_rlinf_alignment.md`](04_rlinf_alignment.md) | 与 RLinf WoVR/embodiment 的 group/worker/channel 对齐。 |
| [`05_cotrain_data_contracts.md`](05_cotrain_data_contracts.md) | message、trajectory shape、sidecar 数据契约。 |
| [`06_sync_checkpoint_metrics.md`](06_sync_checkpoint_metrics.md) | 权重同步、warmup checkpoint bridge、manual checkpoint、metrics namespace。 |
| [`07_validation_matrix.md`](07_validation_matrix.md) | unit、tiny smoke、GPU/LIBERO 验证矩阵。 |

## Main Idea

DreamerVLA 的主线是一个**持续滚动的在线 cotrain 周期**：rollout 不断产生新数据，env 侧用
world model 推进想象轨迹，VLA policy 和 world model/classifier 在同一个 loop 里分别更新，
再各自同步回采样侧。整个周期被拆成四个角色清晰的 group。

| Group | Worker | 是否 FSDP | 职责 |
| --- | --- | --- | --- |
| LearnerGroup | `LearnerWorker` | 否 | 训练 world model 和 classifier/reward model，不碰 VLA。 |
| ActorGroup | `EmbodiedFSDPActor` | 是 | 训练 VLA policy：PPO loss、backward、optimizer step、FSDP 通信。 |
| RolloutGroup | `MultiStepRolloutWorker` | 否 | 普通 HF/BasePolicy 推理副本，`eval/no_grad`，只做 `obs -> action chunk`。 |
| EnvGroup | `RealEnvWorker` / `WMEnvWorker` | 否 | 执行真实 LIBERO 或 latent WMEnv，按 chunk 组装 trajectory。 |

两条关键边界：

- **ActorGroup 与 RolloutGroup 分离**：RolloutWorker 是行为策略推理副本，用于采样；ActorWorker
  是学习中的策略，用于更新参数。即使二者是同一份 VLA 架构，也不能合并。
- **LearnerGroup 只管环境模型**：它训练 world model 和 classifier/reward model，绝不训练 VLA。
  WMEnvWorker 用到的 reward model 就是这个 classifier，主线不再引入额外 verifier。

`EnvWorker` 是基类接口；真实环境由 `RealEnvWorker` 承担，world-model environment 由
`WMEnvWorker` 承担。WMEnvWorker 内部加载 world model 和 classifier，从 LearnerGroup 定期同步
这两者的权重。Ray 层另有一个可选 `ReplayGroup`（`ReplayWorker`），为 WM/classifier 训练和
WMEnv bootstrap 提供数据，但它不是 ActorGroup PPO 的数据通道。

## Training Shape

端到端训练分为四段（详见 [`01_complete_loop.md`](01_complete_loop.md)）：

1. `collect`：用 VLA/OFT 在真实 LIBERO 中采集初始 rollout，写出 hidden sidecar（例如
   `obs_embedding`）和 reward 数据，为 warmup 提供监督。
2. `warmup`：用 collect/precompute 数据训练 world model 和 classifier。
3. `cotrain`：EnvGroup + RolloutGroup 采 chunk-level trajectory，直接经 actor channel 送入
   ActorGroup 做 PPO；LearnerGroup 按节拍更新 WM/cls 并同步给 WMEnvWorker。
4. `eval`：训练信号可以来自 WMEnv trajectory reward，但最终指标仍以真实 LIBERO eval 为准。

主线最重要的形态变化是：**Actor PPO 不再从 replay 采样**。trajectory 由 EnvGroup 在 rollout
内组装好，沿 batch 维切分后直接通过 actor channel 发给 ActorGroup。

## Placement

目标 placement 不假设每张卡角色相同。对 `N` 张 GPU，默认形态是：

```text
GPU0:
  RealEnvWorker + RolloutWorker + LearnerWorker

GPU1..GPU(N-1):
  WMEnvWorker + RolloutWorker + ActorGroup rank
```

这不是固定八卡假设：八卡 = 1 张真实环境卡 + 7 张 WMEnv/Actor 卡；六卡 = 1 + 5。`N=1` 时所有
角色落在 GPU0；`N=0` 是 CPU startup 模式，actor FSDP strategy 为 `none`。ActorGroup 只在 actor
ranks 内部用 FSDP，rank 间的参数 shard、梯度同步和 optimizer state 全部由 FSDP/NCCL 管理。

## Sync Principle

主线有三类相互独立的同步：

1. **ActorGroup 内部**：FSDP/NCCL 负责，不经过 WeightSyncer，不手动复制 rank 权重。
2. **ActorGroup -> RolloutGroup**：每隔 `sync_every` 个 global step，rollout 前从 actor 同步 VLA。
   使用 patch sync——rollout 端持有本地 HF policy 副本，把 actor rank0 发布的 delta apply 到
   本地 `state_dict`。
3. **LearnerGroup -> WMEnvWorker**：LearnerGroup 更新 WM/cls 后，把新版本显式拷给 WMEnvWorker。

## Reward / PPO

主线使用 **trajectory 级别 reward**，advantage 走 group-relative/GRPO 归一化，因此不需要
final obs bootstrap value。final bootstrap value 只属于未来可选的 value-based PPO/GAE 分支
（`requires_bootstrap_value`、`prev_values`、`returns`）；当前主线把它当成 diagnostics 占位，
不参与训练。

ActorGroup 训练 PPO 的关键数据是 `old_logprobs + rewards/dones + forward_inputs`。Actor 用
`forward_inputs` 重新跑当前 VLA 得到 `new_logprobs`，再用 `new/old logprob ratio` 做 PPO
clipped loss。`forward_inputs` 必须完整到能让 Actor 对**同一个 action chunk** 重算 logprob。

## Current Implementation

当前 manual route 已落地为 `ManualCotrainRayRunner`：

```text
experiment=manual_cotrain_ray_oft_backbone_latent
_target_=dreamervla.runners.ManualCotrainRayRunner
```

它创建 `LearnerGroup`、`ActorGroup`、`RolloutGroup`、`RealEnvGroup`、可选 `WMEnvGroup` 和可选
`ReplayGroup`，并已按 chunk-level trajectory contract 连接 Env/Rollout/Actor。当前剩余重点不是
拓扑代码是否存在，而是目标 GPU/LIBERO 机器上完整跑通 async cold-start -> warmup ->
manual cotrain，并完成一次 global step。实现事实见
[`03_current_implementation.md`](03_current_implementation.md)，数据契约见
[`05_cotrain_data_contracts.md`](05_cotrain_data_contracts.md)，验证矩阵见
[`07_validation_matrix.md`](07_validation_matrix.md)。
