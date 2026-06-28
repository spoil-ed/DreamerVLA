# Ray Implementation

状态：目标 Ray/worker 拓扑说明

这份文档说明如何用 Ray worker group 实现 [`01_complete_loop.md`](01_complete_loop.md)
里的目标主线。当前代码还没有完全达到这里的拓扑；实现时以
[`99_manual_notes.md`](99_manual_notes.md) 的用户第一性指导为准。

## 1. Target Worker Graph

目标 Ray 图是四个 group：

```text
LearnerGroup
  LearnerWorker

ActorGroup
  EmbodiedFSDPActor rank(s)

RolloutGroup
  MultiStepRolloutWorker rank(s)

EnvGroup
  EnvWorker base
    RealEnvWorker
    WMEnvWorker
```

职责：

- `LearnerWorker`：非 FSDP，训练 world model 和 classifier/reward model。
- `EmbodiedFSDPActor`：FSDP，训练 VLA，处理 PPO loss、backward、optimizer、FSDP 通信。
- `MultiStepRolloutWorker`：非 FSDP，普通 HF/BasePolicy 推理副本，只负责 no-grad 采样。
- `RealEnvWorker`：真实 LIBERO 环境 worker。
- `WMEnvWorker`：latent world-model env worker，内部有 world model 和 classifier/reward model。

## 2. Placement

目标 placement 按第一张卡和其它卡区分。

对 `N` 张 GPU：

```text
GPU0:
  RealEnvWorker
  MultiStepRolloutWorker
  LearnerWorker

GPU1..GPU(N-1):
  WMEnvWorker
  MultiStepRolloutWorker
  EmbodiedFSDPActor rank
```

这只是默认 profile，不应硬编码。实际 GPU 数、每卡 env slots、Actor world size 和是否启用
真实环境卡，都必须来自 Hydra。

## 3. Channels

目标主循环使用 channel，而不是 driver 手动串联每一步。

核心 channel：

- `env_channel`：EnvWorker 和 RolloutWorker 之间传 obs/action。
- `actor_channel`：EnvWorker 向 ActorGroup 发送 trajectory shard。
- `sync_channel` 或同步接口：ActorGroup -> RolloutGroup，LearnerGroup -> WMEnvWorker。

推荐消息：

```text
ObservationMsg(env_id, task_id, episode_id, step, obs, version)
RolloutResult(actions, prev_logprobs, prev_values, forward_inputs, versions)
Trajectory(actions, rewards, dones, prev_logprobs, forward_inputs, versions, optional values)
WeightPatchMsg(component, version, patch_or_state)
StopMsg(reason)
```

## 4. Global Step

一个 global step 内：

1. ActorGroup 和 RolloutGroup 设置当前 global step。
2. 如果到同步间隔，ActorGroup 把 VLA patch 同步给 RolloutGroup。
3. EnvGroup 启动 `interact`，RolloutGroup 启动 `generate`。
4. EnvWorker 把 obs 发给 RolloutWorker。
5. RolloutWorker 返回 action chunk、old logprob、forward inputs 和版本号。
6. EnvWorker step 真实 LIBERO 或 WMEnv，缓存训练字段。
7. rollout epoch 结束后，EnvWorker 切分 trajectory，发给 ActorGroup。
8. ActorGroup 接收 trajectory，计算 advantage/return，执行 PPO/FSDP 训练。
9. 到 `learner_update_step` 时，LearnerGroup 更新 WM/cls，并同步给 WMEnvWorker。
10. Runner 记录 metrics，按配置 eval/checkpoint。

## 5. EnvWorker

`EnvWorker` 是接口基类，至少需要：

- reset/bootstrap env。
- 接收 action chunk。
- 执行 env step。
- 缓存 rewards/dones/actions/old_logprobs/forward_inputs/versions。
- rollout epoch 结束后组装并切分 trajectory。
- 把 trajectory shards 发到 actor channel。

`RealEnvWorker`：

- 构建真实 LIBERO env。
- 用真实 image/state 作为 obs。
- 保留真实环境锚点和 eval 对齐。

`WMEnvWorker`：

- 加载 world model 和 classifier/reward model。
- 从初始 image 经过 VLA Encoder 得到 latent state。
- 后续在 latent state 中 step。
- 定期从 LearnerGroup 同步 WM/cls。

每个 EnvWorker 可以有多个子 env slots。子 env 接 action，返回 state/image/latent obs；
父 EnvWorker 负责 batch 管理和 trajectory 聚合。

## 6. RolloutWorker

`MultiStepRolloutWorker` 持有普通推理副本：

- 不是 FSDP。
- `eval/no_grad`。
- 从 ActorGroup 定期同步 VLA 权重。
- 输入 obs 或 `obs_embedding`。
- 输出 action chunk、old logprob、forward inputs、版本号。

RolloutWorker 必须把训练所需 action/token/input 放进 `forward_inputs`，因为 ActorGroup
之后要用这些输入重新计算当前 policy 的 logprob。

## 7. ActorGroup

ActorGroup 使用 `EmbodiedFSDPActor`。

每个 Actor rank：

1. 从 actor channel 接收一个或多个 trajectory shard。
2. 沿 batch 维合并 trajectory。
3. 把 `[rollout_epoch * n_chunk_steps, bsz, ...]` 整理成
   `[n_chunk_steps, rollout_epoch * bsz, ...]`。
4. 根据 trajectory 级别 reward 计算 advantage/return。
5. flatten + shuffle。
6. 用 `forward_inputs` 重新跑当前 Actor，得到 `new_logprobs`。
7. 用 `new_logprobs - old_logprobs` 做 PPO clipped loss。
8. 通过 FSDP backward 和 optimizer 更新 VLA。

如果未来切到 value-based PPO/GAE，可以启用 `prev_values`、`returns` 和 final bootstrap
value。当前主线是 trajectory 级别 reward，不需要 final obs bootstrap value。

## 8. LearnerGroup

LearnerGroup 非 FSDP，只负责环境模型：

- world model
- classifier/reward model

它不训练 VLA，也不接管 ActorGroup 的 PPO batch。它按 `learner_update_step` 从配置指定
数据流中训练 WM/cls，然后把新版本同步给 WMEnvWorker。

主线里没有额外 verifier：WMEnvWorker 的 reward model 就是 classifier。

## 9. Sync

ActorGroup 内部同步：

- FSDP/NCCL 管理参数 shard、梯度同步和 optimizer state。
- 不需要手动把某个 Actor rank 的权重复制到其它 Actor rank。

ActorGroup -> RolloutGroup：

- rollout 前同步。
- 目标使用 patch sync。
- Rollout 端本地已有 HF VLA 副本，只 apply patch 到 `state_dict`。

LearnerGroup -> WMEnvWorker：

- LearnerGroup 更新 WM/cls 后同步。
- WMEnvWorker 加载新 world model 和 classifier/reward model state。

## 10. Current Code Gap

当前代码更接近：

```text
ReplayWorker
EnvWorker
RolloutInferenceWorker
LearnerWorker
driver overlap loop
```

目标主线需要补齐或改造：

- `LearnerGroup` 和 `ActorGroup` 分离。
- `EmbodiedFSDPActor` 作为 VLA PPO 训练 worker。
- `MultiStepRolloutWorker.generate` 替代当前单一 `forward_batch` 语义。
- `RealEnvWorker` / `WMEnvWorker` 子类化。
- `EnvWorker.interact` 内部维护 rollout epoch/chunk step 双层循环。
- trajectory 通过 actor channel 直达 ActorGroup，而不是 Actor PPO 依赖 replay sample。
- Actor -> Rollout patch sync。
- Learner -> WMEnv 的 WM/cls 同步。

## 11. Acceptance

最低验收：

- RealEnvWorker 能产生真实 LIBERO trajectory。
- WMEnvWorker 能用同步后的 WM/cls 产生 imagined trajectory。
- RolloutWorker 能为两类 env 输出 action chunk、old logprob 和完整 `forward_inputs`。
- ActorGroup 能接收 trajectory shard，合并、reshape、计算 advantage，并完成 FSDP PPO step。
- LearnerGroup 能更新 WM/cls，并同步到 WMEnvWorker。
- checkpoint 包含 Actor VLA、Learner WM/cls、版本号和 optimizer/scheduler 状态。
- eval 使用真实 LIBERO，并可复现实验配置。
