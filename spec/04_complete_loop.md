# Complete Loop

状态：目标端到端 loop 说明

这份文档按 [`99_manual_notes.md`](99_manual_notes.md) 的第一性指导，把主线从 collect、warmup
讲到 manual cotrain，并说明 trajectory 如何直接进入 ActorGroup 做 PPO。Ray 层 worker graph、
placement、channel、同步和 checkpoint 也在本文中按主线顺序说明。

## 0. Entry

统一训练入口：

```text
python -m dreamervla.train experiment=<name> task=<suite>
```

完整 cold-start 流水线入口：

```text
python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray ...
```

由 `scripts/e2e_coldstart_warmup_cotrain_{ray,noray}.sh` 转发。Runner 负责构建 group、env、
模型、同步器、checkpoint 和 metrics；脚本只保留一条 Hydra 命令，不承载业务逻辑。

`cotrain_engine=async` 时，launcher 先跑一段 warmup-only sync 阶段，把 WM/cls warmup checkpoint
合并成一个 init 文件，再启动 `experiment=manual_cotrain_ray_oft_backbone_latent` 的 Ray async
主线。

## 1. Collect

collect 先用 VLA/OFT 在真实 LIBERO 中采集初始数据，为 WM/cls warmup 提供监督。实现边界：

- 并行派生 worker，每个 worker 负责一部分 task/episode。
- policy 用当前 VLA/OFT 产生 action chunk，真实 LIBERO 执行 action。
- extractor 记录 world model 需要的 hidden sidecar，例如 `obs_embedding`、`lang_emb`。
- reward shard 记录 success、reward、动作、状态和任务信息。

collect 只为 warmup 准备数据，不是最终 cotrain 拓扑。

## 2. Warmup

warmup 训练两个环境侧模型：

1. **World model**：从 `obs_embedding[t] + wm_action[t]` 预测后续 latent。
2. **Classifier/reward model**：判断 latent trajectory 是否成功。

这里的 classifier 就是 WMEnvWorker 用的 reward model；主线不引入额外 verifier。warmup 得到的
WM/cls checkpoint 通过 `learner.init_ckpt`（`{path, components: [world_model, classifier]}`）进入
LearnerGroup，并由 LearnerGroup 在 cotrain 中继续更新。

## 3. Cotrain Groups

cotrain 运行时有四个角色 group（Ray 层 EnvGroup 拆成 `RealEnvGroup`/`WMEnvGroup`，另有可选
`ReplayGroup`）：

| Group | Worker | 职责 |
| --- | --- | --- |
| LearnerGroup | `LearnerWorker(mode=wm_classifier_only)` | 非 FSDP，只更新 world model 和 classifier/reward model。 |
| ActorGroup | `EmbodiedFSDPActor` | FSDP，训练 VLA policy，负责 PPO backward 和 optimizer step。 |
| RolloutGroup | `MultiStepRolloutWorker` | 非 FSDP，`eval/no_grad` 推理副本，只做 `obs -> action chunk`。 |
| EnvGroup | `RealEnvWorker` / `WMEnvWorker` | 执行真实环境或 WMEnv，按 chunk 组装 trajectory。 |

RolloutWorker 是行为策略，用于采样；ActorWorker 是学习中的策略，用于更新 VLA 参数。两者角色
不同，不能合并。

## 4. Placement

默认 `N` 卡目标拓扑（`N=0` 为 CPU startup，`N=1` 全在 GPU0）：

```text
GPU0:
  Real LIBERO EnvWorker
  RolloutWorker
  LearnerWorker

GPU1..GPU(N-1):
  WMEnvWorker
  RolloutWorker
  EmbodiedFSDPActor rank
```

每个 EnvWorker 内部管理多个子 env slots。子 env 接 action，返回 state/image/latent 观测，并把
trajectory 数据交回父 EnvWorker 聚合。

## 5. One Global Step

一个 `global_step` 的主流程是：

1. 给 ActorGroup、RolloutGroup 和 EnvGroup 设置当前 global step/version。
2. 如果 `global_step % sync_every == 0`，ActorGroup 把最新 VLA patch 同步给 RolloutGroup。
3. EnvGroup 和 RolloutGroup 并发运行：EnvWorker 发送 obs，RolloutWorker 返回 action chunk、
   old logprob、forward inputs 和版本号；EnvWorker 执行 LIBERO 或 WMEnv step。
4. rollout epoch 内，EnvWorker 把每个 chunk 切成一个 chunk-level trajectory shard，经 actor
   channel 发给 ActorGroup。
5. ActorGroup 合并 shard，计算 advantage/return，执行 PPO clipped loss 和 FSDP backward。
6. 如果 `global_step % learner_update_step == 0`，LearnerGroup 更新 WM/cls，并把新状态显式同步
   给 WMEnvWorker。
7. Runner 记录 env/rollout/actor/learner/sync/time metrics，并按 `checkpoint_every` 可选 checkpoint。

> 本计划阶段的成功标准是“成功启动并完成一次 global step”。真实 LIBERO eval 是下游质量门，不在
> 短启动目标之内。

## 6. Env Interaction

EnvGroup 有两类 worker：

- `RealEnvWorker`：运行真实 LIBERO，保留真实环境锚点和 eval 对齐。
- `WMEnvWorker`：运行 latent WMEnv，用 world model 推进 latent state，用 classifier/reward model
  给 trajectory reward。

WMEnvWorker 初始化时先从 replay 或离线数据加载初始 `obs_embedding`（必要时还有 `lang_emb`、
`proprio`），用 VLA Encoder 得到 latent state；之后 WMEnvWorker 内部只在 latent state 上推进，
不再每步重走完整真实环境。bootstrap 是 best-effort：replay 为空或缺 key 时回落到 env 配置的
reset 行为。

RolloutWorker 必须能处理两类输入：真实环境的 `obs_embedding`，以及 WMEnv 的 `latent` 观测。

## 7. Rollout Output

每个 chunk step，RolloutWorker 接收当前 obs，调用当前 rollout policy，返回一个 `RolloutResultMsg`：

- `actions`：用于推进环境的 action chunk。
- `prev_logprobs`：rollout policy 对该 chunk 的 old logprob。
- `prev_values`：可选；只有 value-based PPO/GAE 分支需要。
- `forward_inputs`：ActorGroup 重新 forward 当前 Actor 所需的完整输入（至少 `hidden` + `action`）。
- `versions`：policy 版本及可见的 WM/cls 版本。

必须保持的细节：

- 推进环境用 `RolloutResultMsg.actions`。
- 训练缓存用 `forward_inputs["action"]`（与 `actions` 表示同一个 chunk）。

OpenVLA-OFT 路径下，真实 image obs 没有现成 `obs_embedding`：RolloutGroup 用 `OFTRolloutBundle`
把 image 编码成 `obs_embedding`/`lang_emb`，再放进 `forward_inputs`，让 EnvWorker 写入 replay
transition sidecar。这样 ActorGroup 重算 logprob 时，用的是同一批 action/token/input。

## 8. Trajectory Assembly

EnvWorker 对每个 chunk 产出一个 chunk-level `TrajectoryShard`（leading shape `[1, 1, ...]`）：

```text
actions:        [1, 1, chunk, action_dim]
rewards:        [1, 1, chunk]
dones:          [1, 1, chunk]
prev_logprobs:  [1, 1]
forward_inputs: [1, 1, ...]
versions:       [1, 1]
```

一个 chunk 中途 terminated/truncated 时，`actions` 仍保留**完整** sampled chunk 供 PPO 评估，但
环境只执行到 terminal 为止：未执行的 reward 填 `0.0`，未执行的 done 置 terminal，未执行的
transition 不写 replay，并在 episode 边界 reset，不在同一 chunk 内跨 episode 继续。

ActorGroup 端通过 `collate_trajectory_shards` 沿 batch 维（`dim=1`）拼接成 `TrajectoryBatch`：

```text
actions:        [T, B, chunk, action_dim]
rewards:        [T, B, chunk]
dones:          [T, B, chunk]
prev_logprobs:  [T, B]
forward_inputs: [T, B, ...]
```

## 9. Actor PPO

ActorGroup 从 actor channel 接收 trajectory shard，collate 成 `TrajectoryBatch`，然后：

1. 由 trajectory 级别 reward 计算 return：对 time 维和任意 chunk trailing 维求和，得到 `[B]`。
2. 用 group-relative/GRPO 把 return 归一化成 advantage（按 `group_size`）。
3. 用 `forward_inputs` 重新跑当前 Actor 得到 `new_logprobs`：

```text
current Actor + forward_inputs(mode=evaluate, hidden, action, ...) -> new_logprobs
ratio = exp(new_logprobs - old_logprobs)
loss  = PPO clipped surrogate(ratio, advantage; clip_ratio_low, clip_ratio_high) - entropy_coef * entropy
```

Actor 强制要求 `batch.actions.ndim == 4`（chunk-level），不会把 chunk 展平成单步训练。Actor 是
FSDP，每个 rank 只处理自己的 microbatch；gradient accumulation 结束后由 FSDP 负责参数 shard、
梯度同步和 optimizer state 更新。

关键约束：`forward_inputs` 必须足够完整。如果未来 VLA encoder 也参与训练，只存 detached
embedding 会断开训练路径；只有 encoder 冻结时才可以只存 latent/embedding。

主线 reward 是 trajectory 级别，因此不需要 `prev_values`、`returns` 和 final bootstrap value；
只有未来切到 `actor_critic + gae` 才启用它们。

## 10. LearnerGroup Update

LearnerGroup 不训练 VLA，只训练 world model 和 classifier/reward model。每隔
`learner_update_step` 个 global step，它从配置指定的数据流（collect 数据、真实 rollout、
WMEnv rollout 或单独的 WM/cls 缓存）取样更新 WM/cls。这个数据流不替代 ActorGroup 的 PPO
trajectory channel。更新完成后，LearnerGroup 把最新 WM/cls 状态显式同步给 WMEnvWorker。

## 11. Sync / Checkpoint / Metrics

同步分三类：

- ActorGroup 内部：FSDP/NCCL 管理，不手动复制。
- ActorGroup -> RolloutGroup：rollout 前 patch sync VLA 权重。
- LearnerGroup -> WMEnvWorker：同步 world model 和 classifier/reward model。

需要记录的 metrics namespace：`env/`、`rollout/`、`actor/`、`train/`（learner WM/cls loss）、
`sync/`、`replay_buffer/`、`time/`。manual checkpoint 至少包含 ActorGroup VLA、LearnerGroup
WM/cls、global step、版本号和 Hydra resolved config。

## 12. Eval

训练信号可以来自 WMEnv trajectory reward，但最终指标仍以真实 LIBERO eval 为准。Eval 使用当前
Actor/VLA 权重，在真实 LIBERO 上跑固定 task suite，记录 success rate、episode return 和视频。
