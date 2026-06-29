# RLinf Alignment

状态：current/target 混合

DreamerVLA manual cotrain 有意沿用 RLinf WoVR/embodiment 在 group、worker、channel 和
rollout-loop 边界上的组织方式。本文记录对齐目标和当前 DreamerVLA 映射。参考实现位于
`/mnt/data/spoil/workspace/RLinf`，最贴近的是：

- `rlinf/runners/embodied_runner.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/data/embodied_io_struct.py`
- `examples/embodiment/config/wan_libero_goal_grpo_openvlaoft_4567.yaml`

实现 DreamerVLA cotrain 前，应先理解 RLinf 如何拆分这些角色，并尽量先跑通/复现 WoVR 的短流程
启动。

## Reference Scope

相关 RLinf 概念：

| RLinf role | 职责 |
| --- | --- |
| `ActorGroup` | FSDP policy training、optimizer step、backward、参数/梯度同步。 |
| `RolloutGroup` | `no_grad` HF/BasePolicy 推理副本，在 rollout 边界从 actor 同步。 |
| `EnvGroup` | real 或 imagined env stepping、obs/action 交换、trajectory assembly。 |
| embodied rollout structs | chunk-level action/logprob/reward/forward-input 记录。 |

DreamerVLA 在此之上多加一个 group：

| DreamerVLA role | 职责 |
| --- | --- |
| `LearnerGroup` | 只更新 world-model 和 classifier/reward-model，绝不训练 VLA policy。 |

## Current Mapping

| DreamerVLA component | RLinf-style role |
| --- | --- |
| `ManualCotrainRayRunner` | manual embodied runner / control loop |
| `WorkerGroup` | Ray group wrapper |
| `Channel` | observation / rollout-result / trajectory transport |
| `EmbodiedFSDPActor` | ActorGroup worker |
| `MultiStepRolloutWorker` | RolloutGroup worker |
| `RealEnvWorker` / `WMEnvWorker` | EnvGroup workers |
| `TrajectoryShard` / `TrajectoryBatch` | embodied trajectory payload |
| `LearnerWorker(mode=wm_classifier_only)` | DreamerVLA-only LearnerGroup |

## Boundary Rules

- ActorGroup 和 RolloutGroup 始终分离，即使共用同一 policy 架构。
- RolloutGroup 永不做 optimizer step。
- manual route 下 ActorGroup 不从 replay 采 PPO batch，它消费 EnvGroup 的 trajectory shard。
- LearnerGroup 不训练 VLA policy（`wm_classifier_only` 模式校验组件只能是 `world_model` 和
  `classifier`，并拒绝 FSDP）。
- WMEnvWorker 拥有 imagined env stepping，但它的 world model/classifier 状态来自 LearnerGroup。
- Replay 仍用于 WM/classifier 学习和 WMEnv bootstrap；它不是 Actor PPO 的 trajectory channel。

## Chunk-Level Semantics

manual route 遵循 chunk-level rollout 语义：

```text
RolloutWorker 采样一个 action chunk
EnvWorker 执行该 chunk 直到 terminal/truncated 或 chunk 末尾
EnvWorker 产出一个 chunk-level TrajectoryShard
ActorWorker 用 forward_inputs 对同一个 action chunk 重算 logprob
```

如果 episode 在 chunk 中途结束：剩余 reward slot 填 `0.0`，剩余 done slot 置 terminal，未执行的
transition 不写 replay，并在 episode 边界 reset；但 sampled action chunk 仍完整保留，供 policy
logprob 评估。这与 RLinf 把一个 chunk 记成一个 `ChunkStepResult`（而非展开成多个 actor step）的
做法一致。

## Current Divergence From RLinf

DreamerVLA 有意保持 single-node、Hydra-owned。它不追求多机 RLinf 部署、自动 VRAM sizing、
vLLM/SGLang rollout 服务或通用 placement 模式。manual route 继续使用 DreamerVLA 自己的 builder、
OpenVLA-OFT sidecar 处理、LIBERO wrapper、`OnlineReplay` 和 warmup checkpoint bridge，而不是直接
import RLinf 代码。
