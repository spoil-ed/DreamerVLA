# Complete Loop

状态：目标端到端 loop 说明

这份文档按 [`99_manual_notes.md`](99_manual_notes.md) 的第一性指导描述目标主线。
它说明如何从 collect、warmup 进入 cotrain，并把 trajectory 直接送入 ActorGroup 做 PPO。

## 0. Entry

统一入口仍然是：

```text
python -m dreamervla.train experiment=<name> task=<suite>
```

Runner 负责构建 group、env、模型、同步器、checkpoint 和 metrics。脚本只保留单条 Hydra
命令，不承载业务逻辑。

## 1. Collect

collect 先用 VLA/OFT 在真实 LIBERO 中采集初始数据，为 WM/cls warmup 提供监督。

实现边界：

- 并行派生 worker，每个 worker 负责一部分 task/episode。
- policy 使用当前 VLA/OFT 产生 action chunk。
- 真实 LIBERO 环境执行 action。
- extractor 记录 world model 需要的 hidden sidecar，例如 `obs_embedding`。
- reward shard 记录 success、reward、动作、状态和任务信息。

collect 阶段只为 warmup 准备数据，不是最终 cotrain 拓扑。

## 2. Warmup

warmup 训练两个环境侧模型：

1. World model：从 `obs_embedding[t] + wm_action[t]` 预测后续 latent。
2. Classifier/reward model：判断 latent trajectory 是否成功。

这里的 classifier 就是 WMEnvWorker 使用的 reward model；主线不再引入额外 verifier。

warmup 后得到的 WM/cls checkpoint 会进入 LearnerGroup，并由 LearnerGroup 在 cotrain 中继续更新。

## 3. Cotrain Groups

cotrain 运行时有四个 group：

| Group | 包含 worker | 职责 |
| --- | --- | --- |
| LearnerGroup | LearnerWorker | 非 FSDP，训练 world model 和 classifier/reward model。 |
| ActorGroup | EmbodiedFSDPActor | FSDP，训练 VLA policy，负责 PPO backward 和 optimizer step。 |
| RolloutGroup | MultiStepRolloutWorker | 非 FSDP，普通推理副本，只做 `obs -> action chunk`。 |
| EnvGroup | RealEnvWorker / WMEnvWorker | 执行真实环境或 WMEnv，组装 trajectory。 |

ActorGroup 和 RolloutGroup 是两个不同角色：RolloutWorker 是行为策略推理副本，用于采样；
ActorWorker 是学习中的策略，用于更新 VLA 参数。

## 4. Placement

默认 `N` 卡目标拓扑：

```text
GPU0:
  Real LIBERO EnvWorker
  RolloutWorker
  LearnerGroup / LearnerWorker

GPU1..GPU(N-1):
  WMEnvWorker
  RolloutWorker
  ActorGroup / EmbodiedFSDPActor rank
```

每个 EnvWorker 内部可以管理多个子 env slots。子 env 接受 action，返回 state/image/latent
观测，并把 trajectory 数据交回父 EnvWorker 聚合。

## 5. One Global Step

一个 `global_step` 的主流程是：

1. 设置 ActorGroup 和 RolloutGroup 的 global step/version。
2. 如果达到同步间隔，ActorGroup 把最新 VLA 权重同步给 RolloutGroup。
3. EnvGroup 和 RolloutGroup 并发运行：EnvWorker 发送 obs，RolloutWorker 返回 action chunk、
   old logprob、forward inputs 和版本信息。
4. EnvWorker 执行真实 LIBERO 或 WMEnv step，记录 rewards/dones/actions/old_logprobs/
   forward_inputs/versions。
5. rollout epoch 结束后，EnvWorker 把缓存结果切成 trajectory shards，通过 actor channel
   发送给 ActorGroup。
6. ActorGroup 合并 trajectory，计算 advantage/return，执行 PPO clipped loss 和 FSDP backward。
7. 每固定 `learner_update_step` 个 global step，LearnerGroup 更新 WM/cls，并同步给 WMEnvWorker。
8. Runner 按配置执行 eval、checkpoint 和 metrics 记录。

## 6. Env Interaction

EnvGroup 有两类 worker：

- `RealEnvWorker`：运行真实 LIBERO，用于保留真实环境锚点。
- `WMEnvWorker`：运行 latent WMEnv，用 world model 推进 latent state，用 classifier/reward
  model 给 trajectory reward。

WMEnvWorker 初始化时先从离线数据或真实环境数据加载初始 image，再用 VLA Encoder 得到
latent state。之后 WMEnvWorker 内部只在 latent state 上推进，不再每步重新走完整真实环境。

RolloutWorker 必须能处理两类输入：

- 真实环境 obs/image。
- WMEnv 提供的 `obs_embedding` 或 latent obs。

## 7. Rollout Output

每个 chunk step，RolloutWorker 接收当前 obs，调用当前 rollout policy，返回：

- `actions`：用于推进环境的 action chunk。
- `prev_logprobs`：rollout policy 对该 action 的 old logprob。
- `prev_values`：可选；只有 value-based PPO/GAE 分支需要。
- `forward_inputs`：ActorGroup 训练时重新 forward 当前 Actor 所需的完整输入。
- `versions`：rollout policy 版本。

必须保持的细节：

- 推进环境使用 `rollout_result.actions`。
- 训练缓存使用 `rollout_result.forward_inputs["action"]` 或等价字段。

这样 ActorGroup 重新计算 logprob 时，能保证使用的是同一批 action/token/input。

## 8. Trajectory Assembly

EnvWorker 在 rollout epoch 内维护 trajectory buffer。核心字段是：

- `actions`
- `rewards`
- `dones`
- `prev_logprobs`
- `forward_inputs`
- `versions`
- optional `prev_values`

采样结束后，EnvWorker 把 list stack 成时间优先 trajectory：

```text
actions:        [T, B, ...]
prev_logprobs:  [T, B, ...]
rewards:        [T, B, ...]
forward_inputs: [T, B, ...]
dones:          [T, B, ...]
```

如果启用 value-based PPO/GAE，`prev_values` 和 `dones` 可以带 `T + 1` bootstrap 维度。
但主线使用 trajectory 级别 reward，不需要 final obs bootstrap value。

EnvWorker 再按 `actor_split_num` 沿 batch 维切分 trajectory，让每个 Actor rank 收到自己的 shard。

## 9. Actor PPO

ActorGroup 每个 rank 从 actor channel 接收若干 trajectory shard，并沿 batch 维合并。
Actor PPO 通过 actor channel 直接接收 trajectory，不依赖 replay sample。

合并后先把原始形状：

```text
[rollout_epoch * n_chunk_steps, bsz, ...]
```

整理为：

```text
[n_chunk_steps, rollout_epoch * bsz, ...]
```

这样 advantage 可以沿时间维计算。

主线 reward 是 trajectory 级别 reward，因此 advantage 可以按 GRPO/group reward 归一化。
如果以后启用 `actor_critic + gae`，才需要 `prev_values`、`returns` 和 final bootstrap value。

训练前 ActorGroup 把 `[T, B, ...]` flatten 成 `[T * B, ...]`，再 shuffle。随后 Actor 使用
`forward_inputs` 重新运行当前 VLA：

```text
current Actor + forward_inputs -> new_logprobs
```

PPO ratio：

```text
ratio = exp(new_logprobs - old_logprobs)
```

clipped loss 使用标准 PPO surrogate：

```text
max(-adv * ratio, -adv * clip(ratio, 1 - clip_low, 1 + clip_high))
```

Actor 是 FSDP，所以每个 Actor rank 只处理自己的 microbatch。gradient accumulation 结束后，
FSDP 负责参数 shard、梯度同步和 optimizer state 更新。

关键约束：`forward_inputs` 必须足够完整，能让 ActorGroup 对同一 action 重新计算当前 VLA
logprob。如果 VLA encoder 也参与训练，只存 embedding 会断开训练路径；只有 encoder 冻结时，
才可以只存 latent/embedding。

## 10. LearnerGroup Update

LearnerGroup 不训练 VLA。它只训练：

- world model
- classifier/reward model

每固定 `learner_update_step` 个 global step，LearnerGroup 从配置指定的数据流中取样更新
WM/cls。这个数据流可以来自 collect 数据、真实 rollout、WMEnv rollout 或单独的 WM/cls
训练缓存，但它不替代 ActorGroup 的 PPO trajectory channel。

LearnerGroup 更新完成后，把最新 WM/cls 同步给 WMEnvWorker。

## 11. Sync / Checkpoint / Metrics

同步分三类：

- ActorGroup 内部：FSDP/NCCL 管理，不手动复制。
- ActorGroup -> RolloutGroup：rollout 前同步 VLA 权重，目标使用 patch sync。
- LearnerGroup -> WMEnvWorker：同步 world model 和 classifier/reward model。

必须记录的 metrics：

- env：steps、episodes、success、real-env/WM-env 分布。
- rollout：action chunks、old logprob stats、policy version。
- actor：PPO loss、ratio、clipfrac、advantage、grad norm、FSDP step。
- learner：WM loss、classifier loss/F1、WM/cls version。
- sync：actor->rollout version、learner->WMEnv version、sync time。
- time：env、rollout、actor、learner、sync wait。

checkpoint 至少包含 ActorGroup VLA、LearnerGroup WM/cls、optimizer/scheduler、global step、
版本号和 Hydra resolved config。

## 12. Eval

训练信号可以来自 WMEnv trajectory reward，但最终指标仍以真实 LIBERO eval 为准。

Eval 使用当前 Actor/VLA 权重，在真实 LIBERO 上跑固定 task suite，记录 success rate、
episode return 和视频。
