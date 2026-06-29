# User First-Principles Guidance

状态：用户手写第一性指导区

这份文件用于记录用户对 DreamerVLA architecture 的第一性判断：目标拓扑、抽象边界、
实现取舍、优先级和不可破坏约束。它不是当前代码事实表；凡是尚未实现的内容，应明确写成
目标方案或设计假设。

使用规则：

- 禁止 Codex、Claude 或其它 AI assistant 在没有用户明确指令时移动、改写、压缩或删除
  本文件中的用户手写内容。
- AI assistant 只能在用户要求时补充代码现状、风险、遗漏点或 `Current vs Target` 差异；
  补充内容不得覆盖、替换或重排用户的第一性指导。
- 当这里与 `00_overview.md`、`01_complete_loop.md`、`02_ray.md` 不一致时，先把差异记录到
  `Current vs Target`，再决定是否更新主文档或代码。
- 确认后的内容再整理进主文档；未确认内容保留在这里。

## First-Principles Direction

在这里写用户的第一性指导。

## Cotrain Target Draft

LearnerGroup
  LearnerWorker
  不是 FSDP
  用于训练 WorldModel Classifier

ActorGroup:
  EmbodiedFSDPActor
  是 FSDP
  用于训练 VLA
  有 optimizer / grad / backward / FSDP 通信

RolloutGroup:
  MultiStepRolloutWorker
  不是 FSDP
  是普通 HuggingFace/BasePolicy 推理副本
  eval/no_grad
  只负责 obs -> action chunk
  定期从 Actor 同步权重

EnvGroup:
  EnvWorker，实际上使用子类 RealEnvWorker WMEnvWorker。基类主要定义接口
  不是 FSDP
  内部加载 WMEnv
  WMEnvWorker 里有 world model + classifier/reward model, 从 LearnerGroup 中定期同步权重
  负责 imagined env step

cotrain 是一个持续滚动的在线训练周期：rollout 产生新数据，replay 缓存这些数据，
LearnerGroup 更新 WM/cls；ActorGroup 更新 VLA；二者分别同步到 WMEnv 和 RolloutWorker。

一次 cotrain 周期可以按自然语言理解为：

1. 模型先启动 EnvWorker，每张卡部署一份 EnvWorker。EnvWorker 负责调用环境，可以是真实 rollout 环境或者是 WM 环境。
   在具体实现中，我们只采用 1 份 EnvWorker 作为真实 rollout 环境，剩下卡作为 WM 环境。
2. 每轮 global_step 开始前，Rollout 从 Actor 同步权重。方案
```
有两层同步：
1. ActorGroup 内部
    Actor 是 FSDP 训练。8 张卡上 8 个 Actor rank 共同训练同一个模型，梯度同步、参数 shard 更新由 FSDP/NCCL
    管，不靠 LoRA，也不需要额外手动把 1 卡 Actor 拷到其他 Actor。
2. Actor -> Rollout
    这是 WeightSyncer 做的。默认 Wan 配置用的是 patch_syncer：
    - rollout 端先加载一份 HF VLA；
    - actor 端 FSDP 训练；
    - 每轮 rollout 前，runner 调 rollout.sync_model_from_actor() 和 actor.sync_model_to_rollout()；
    - actor rank0 把更新后的权重广播给所有 rollout ranks；
    - rollout 直接把收到的 patch apply 到本地 hf_model.state_dict()。
```
3. 每个 EnvWorker 会并发多个子 EnvWorker，每个子 EnvWorker 都是接受动作，返回状态（state or image）
4. 对于真实 rollout EnvWorker，每个 slot 记录当前观测、episode buffer、task id、episode id 和使用的模型版本。
如果某个 episode 结束，EnvWorker 会把整条 episode 写入 replay，并 reset 对应 env slot 开始下一条 episode。EnvWorker 把 policy action 转成环境动作，调用真实 LIBERO step，拿到下一帧观测、reward、done 和 info，然后把本步 transition 追加到对应 episode buffer。如果某个 episode 结束，EnvWorker 会把整条 episode 写入 replay，并 reset 对应 env slot开始下一条 episode。
5. 对于 WMEnvWorker，把 VLA 分为 Encoder 和 Actor 部分；先从离线数据 load 最开始的 init_state 的 image，
再用 Encoder 去嵌入为 latent state。之后在 WMEnvWorker 中就只用 latent state 进行推理了。
6. RolloutWorker 用当前 VLA/OFT policy 批量推理动作，并同时产出 world model 需要的 `obs_embedding`，以及可选的 `lang_emb`。另外，在 RolloutWorker 部分，需要能够接受 obs_embedding 作为 WMEnvWorker 的输入
7. 每固定 learner_update_step 步（在 global_step 维度），driver 触发一次 cotrain update，更新 WorldModel 和 cls；
rollout 继续使用当前已同步的 policy 版本采样。learner update 结束后，driver 收集训练 metrics，发布新的 policy/world model/classifier 版本。RolloutWorker 在安全边界上拉取新权重，replay 继续记录每条数据对应的 meta data。
8. runner 按配置周期保存 checkpoint 和 metrics，然后继续下一轮 rollout 与 learner update，直到达到 step、episode、时间或外部停止条件。

RolloutWorker = 行为策略推理，用来采样数据
ActorGroup    = 学习中的策略，用来更新参数
for global_step:
    设置 actor/rollout 的 global_step
    如果到 sync interval: actor 权重同步到 rollout
    并发启动:
        env.interact(...)       # 环境侧收动作、step、收集轨迹
        rollout.generate(...)   # policy 推理，给 env 发动作和 logprob
        reward.compute_rewards  # 当前 Wan 配置通常没有外部 reward worker
    actor.recv_rollout_trajectories(...)
    actor.compute_advantages_and_returns()
    actor.run_training()
    可选 eval
    可选 save checkpoint
    记录 env/rollout/train/time metrics
GPU0:
EnvWorker rank
  └── 1 个 LIBEROEnv
        ├── 1 个 Libero Env
        └── batched 管理 8 个并行 env 状态

GPU1-7:
EnvWorker rank
  └── 1 个 WMEnv
        ├── 1 个 WM
        ├── 1 个 classifier/reward model
        └── batched 管理 8 个并行 env 状态

内层 rollout loop
EnvWorker._run_interact_once 是采样主循环，它做两层循环：

for rollout_epoch in 16:
    reset/bootstrap env，发送初始 obs 给 RolloutWorker

    for chunk_step in max_steps_per_rollout_epoch / num_action_chunks:
        RolloutWorker 根据当前 obs 采样一个 action chunk
        env worker 接收 action/logprob/value/forward_inputs
        env 执行 chunk_step
        记录 rewards/dones/logprobs/actions/forward_inputs
        发送新 obs 给 RolloutWorker

    额外请求一次 final obs 的 bootstrap value
    flush video / 更新 reset ids

## Data Flow



## Current vs Target

## Open Questions

在这里记录需要讨论的问题。

## Decisions
