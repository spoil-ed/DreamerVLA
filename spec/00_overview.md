# Architecture Overview

状态：目标主线架构入口

`spec/` 是 DreamerVLA 当前 architecture 的根目录。本目录的主线说明以
[`99_manual_notes.md`](99_manual_notes.md) 的用户第一性指导为准。
`99_manual_notes.md` 不是当前代码事实表，但它定义目标架构方向；除非用户明确要求，
AI assistant 不得移动、改写或删除其中的用户手写内容。
它是受保护的第一性来源。

| 文件 | 作用 |
| --- | --- |
| [`00_overview.md`](00_overview.md) | 说明目标主线架构、核心 group、训练形态。 |
| [`01_complete_loop.md`](01_complete_loop.md) | 说明端到端 loop：collect、warmup、cotrain、PPO、eval。 |
| [`02_ray.md`](02_ray.md) | 说明 Ray/worker 拓扑、placement、通信和当前代码差异。 |
| [`99_manual_notes.md`](99_manual_notes.md) | 用户第一性指导区，优先级高于整理后的主文档。 |

## Main Idea

目标 cotrain 主线拆成四个 group：

| Group | Worker | 是否 FSDP | 职责 |
| --- | --- | --- | --- |
| LearnerGroup | LearnerWorker | 否 | 训练 world model 和 classifier/reward model。 |
| ActorGroup | EmbodiedFSDPActor | 是 | 训练 VLA policy，执行 PPO backward、optimizer step 和 FSDP 通信。 |
| RolloutGroup | MultiStepRolloutWorker | 否 | 持有普通 HuggingFace/BasePolicy 推理副本，`eval/no_grad` 生成 action chunk。 |
| EnvGroup | RealEnvWorker / WMEnvWorker | 否 | 执行真实 LIBERO 或 latent WMEnv，收集 trajectory。 |

`EnvWorker` 是基类接口；真实环境由 `RealEnvWorker` 承担，world-model environment 由
`WMEnvWorker` 承担。`WMEnvWorker` 内部加载 world model 和 classifier；这里的 classifier
就是 reward model，没有额外 verifier。

## Training Shape

端到端训练分为四段：

1. `collect`：用 VLA 在真实 LIBERO 中采集初始 rollout，保存 hidden sidecar 和 reward 数据。
2. `warmup`：用 collect/precompute 数据训练 world model 和 classifier。
3. `cotrain`：EnvGroup + RolloutGroup 采 trajectory；ActorGroup 直接用 trajectory 训练 PPO；
   LearnerGroup 更新 WM/cls，并同步给 WMEnvWorker。
4. `eval`：最终效果仍以真实 LIBERO eval 为准。

核心变化是：Actor PPO 不再依赖 replay sample。trajectory 由 EnvGroup 组装后直接通过
actor channel 发送给 ActorGroup。

## Placement

目标 placement 不假设每张卡完全一样。

对 `N` 张 GPU，默认形态是：

```text
GPU0:
  RealEnvWorker + RolloutWorker + LearnerGroup

GPU1..GPU(N-1):
  WMEnvWorker + RolloutWorker + ActorGroup rank
```

这不是固定八卡假设。八卡时是 1 张真实环境卡 + 7 张 WMEnv/Actor 卡；六卡时是
1 张真实环境卡 + 5 张 WMEnv/Actor 卡。

ActorGroup 只在 actor ranks 内部使用 FSDP。Actor rank 之间的参数 shard、梯度同步和
optimizer state 由 FSDP/NCCL 管理，不靠手动复制。

## Sync Principle

有三类同步：

1. ActorGroup 内部同步：FSDP 负责，不通过 WeightSyncer。
2. ActorGroup 到 RolloutGroup：rollout 前从 actor 同步 VLA 权重。目标方案使用 patch sync，
   rollout 端持有本地 HF policy 副本，把 actor rank0 发来的 patch apply 到本地 `state_dict`。
3. LearnerGroup 到 WMEnvWorker：LearnerGroup 更新 world model 和 classifier 后，把新版本
   同步给 WMEnvWorker。

## Reward / PPO

主线使用 trajectory 级别 reward，不需要 final obs bootstrap value。final bootstrap value
只属于 value-based PPO/GAE 路线；如果未来切到 actor-critic/GAE，可以作为可选分支加入。

ActorGroup 训练 PPO 的关键数据是：

```text
old_logprobs + rewards/dones + forward_inputs + optional values
```

ActorGroup 重新用 `forward_inputs` 跑当前 VLA，得到 `new_logprobs`，再用
`new/old logprob ratio` 做 PPO clipped loss。

## Current Code Gap

当前代码中已有 Ray worker、replay、online cotrain 和部分 WMEnv 基础能力，但主线目标还需要
从旧的 driver-loop/replay-heavy 形态迁移到：

```text
EnvWorker.interact
RolloutWorker.generate
ActorGroup.recv_rollout_trajectories
LearnerGroup.update_wm_and_classifier
```

实现时应优先对齐 `99_manual_notes.md`，再更新代码。
