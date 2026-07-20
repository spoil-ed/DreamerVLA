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
- 当这里与 `00_overview.md`、`04_complete_loop.md`、`02_naming.md` 不一致时，先把差异记录到
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

### Cotrain、运行目录与恢复的不可回归约束

以下规则是已经确认的主线行为。后续实现、重构、进度条调整和实验脚本修改不得在没有
用户明确指令的情况下改变这些语义。

#### 1. 主线 cotrain 是失败轨迹条件化的 imagined-only RL

一次 cotrain `global_step` 的主线顺序是：

1. 从当前 policy 收集完整的 real trajectories；
2. 将当前真实轨迹追加到历史 replay，并保留 episode 成败标签；
3. 只从失败 episode 中选择首帧作为 imagined rollout 起点；当前和历史失败都可使用，同一
   失败轨迹允许重复采样；
4. 使用冻结的 world model rollout，并使用冻结的 classifier 提供 imagined reward；
5. ActorGroup 只使用 imagined trajectories 执行 advantage 和 PPO/actor update；
6. 按配置执行 evaluation、checkpoint 和 metrics 记录。

主线模式不执行 encoder real-SFT，不重新编码真实轨迹，不更新 world model，不更新
classifier。encoder、world model 和 classifier 均冻结，只有 ActorGroup 的 actor 参数分区
接受 imagined PPO 更新。主线 selector 是失败 episode 的最开始帧，配置必须为以后增加
endpoint、window 或 classifier-guided selector 保留扩展空间。如果当前和历史 replay 都没有
失败 episode，本 global step 明确跳过 imagined rollout 和 PPO，不得回退到成功轨迹。

`openvla_libero_aggressive` 是用户明确选择的 opt-in 对照实验：它保持 WM、classifier 和
encoder 冻结，但把 selector 改为所有 episode 的首帧。它不改变 `openvla_libero` 的上述主线
语义，也不得成为失败池为空时的隐式 fallback。

#### 1.1 最小 imagined-success SFT 训练信号实验

`openvla_libero_success_sft_probe` 是一次性诊断，不是新的主线训练方案。它只加载基础 VLA
和已有 WM/CLS checkpoint，因果顺序固定为：

1. 收集一个 real episode，仅用于向 replay 提供一个 episode-start initial condition；
2. 冻结 encoder、world model 和 classifier，从该起点生成 128 条短 imagined trajectories；
3. 使用 classifier checkpoint 自带的 threshold 选择至少一个时刻达到成功阈值的完整轨迹；
4. ActorGroup 不计算 advantage、不执行 PPO，只对这些成功轨迹中 `loss_mask` 有效的 action
   decisions 最小化行为 action-token 的负 log-likelihood；
5. 在现有 policy KL transaction 内提交一次 actor optimizer step，并写入完整 checkpoint；
6. 自动检查成功轨迹数、有效 SFT 样本数、正且有限的梯度范数、已提交 optimizer step，以及
   policy 初始/最终 hash 确实不同。任一条件不满足时实验以失败退出。

该诊断只回答“现有 WM/CLS 产生的数据能否向 actor 提供可执行的梯度信号”，不能单独证明
real LIBERO success rate 会提高。CLS 没选出成功轨迹时不得降低 threshold、不得把失败轨迹
混入 SFT，也不得伪造一次零梯度 optimizer step。

初始 evaluation 可以发生在第一个 `global_step` 之前。各 Group/Worker 可以在 setup 阶段提前
创建，但“worker 已经创建”不等于“对应阶段正在执行”。因此 real rollout 阶段只显示 real
rollout 是正确行为；WMEnvWorker 此时等待阶段屏障，进入第 6 步后才显示独立的 imagined
rollout。禁止仅为了让两个进度条同时出现而把上述因果流程误改为 real/imagined 并发。

#### 2. Real rollout 和 evaluation 的主进度必须按完整轨迹统计

- real rollout 的主进度分子是所有 `real_env` worker 已完成 trajectory 数之和，分母是配置的
  real trajectory 总目标；单位为 `trajectory`。
- evaluation 的主进度分子是所有 `eval_env` worker 已完成 episode 数之和，分母是配置的
  evaluation episode 总数。这里一个 episode 是一条完整评测轨迹。
- `chunks`、action chunk callback 次数、环境 step 数、worker callback 序号和 worker finished
  状态都不得作为上述主进度分子，也不得让主进度条提前前进。
- chunks 只允许显示在 status/diagnostic 字段中。success rate 必须使用
  `successes / completed`，不能使用 chunks 作为分母。
- imagined rollout 是 real collection 和失败起点选择后的独立后续阶段，使用独立进度条，不能
  混入 real rollout 或 evaluation 的主进度。

如果日志出现 `completed=0` 但 evaluation 主进度已经大于 0，或 real rollout 主单位仍是
`chunk/s`，说明运行的是旧实现或进度口径已经回归，不能把该日志当作当前正确行为。

#### 3. 一个 invocation 只拥有一个浅层 run root

默认 run root 必须是：

```text
<output_root>/<run.name>/<YYYYMMDD_HHMMSS>/
```

同一层级包含：

```text
checkpoints/  wandb/  tensorboard/  logs/  video/  diagnostics/  .hydra/
```

禁止再次嵌套重复的任务名、`wm/`、`classifier/`、`wmcls_cotrain/`、`log/wandb/` 或其它路线
目录。旧日志若仍写入 `log/wandb/` 或 checkpoint 仍写入 `ckpt/`，说明运行的不是当前统一
布局实现。

#### 4. Resume 必须是真正的原地继续训练

- 所有当前可训练实验都必须接受显式 resume path；它可以指向 run root、checkpoint 目录或
  checkpoint 文件。
- resume 后必须继续使用 checkpoint 所属的原 run root，不得创建新 timestamp 目录，也不得
  把新的 W&B、TensorBoard、Hydra、视频或 checkpoint 产物散落到另一个目录。
- 新 checkpoint 统一写入 `checkpoints/`；可以兼容读取历史 `ckpt/`，但不能继续产生新的
  legacy 布局。
- checkpoint 必须保存该路线继续训练所需的模型、optimizer、global step/epoch、best metric、
  classifier threshold 和 RNG 等持久状态。只加载模型权重然后从第 0 步重开不属于 resume。
- Replay 是临时运行态，不得写入 cotrain checkpoint，也不恢复 replay sampling cursor；历史
  checkpoint 中的 replay 字段一律忽略。Cotrain 只支持在完整 global-step 边界 resume，恢复后
  创建新的内存 replay，并按当前 Hydra 实验的数据流从新一轮真实轨迹开始积累或替换。
- `failure_imagined_rl` resume 后不保留历史 failure anchors；如果新一轮真实轨迹没有失败样本，
  按现有语义跳过该步 imagined policy update。
- 所有训练 checkpoint 都平铺在唯一的 `checkpoints/`：始终维护 `latest.ckpt`，并按
  `epoch=<完成 epoch>-<metric>=<value>.ckpt` 保留 Hydra 配置的 top-k；不得创建 step、
  component 或 route 子目录。
- HF 导出必须显式开启，且只写到 run root 下与 `checkpoints/` 同级的
  `checkpoint_hf/`。
- eval run root 固定为 `<output_root>/eval/<任务名>/`，不增加 timestamp 层，也不创建
  checkpoint 输出目录。eval 输入为目录时读取其中的 `checkpoints/latest.ckpt`，具体
  `.ckpt` 文件仍直接兼容。

#### 5. 修改这些行为时必须保留回归证据

涉及以上规则的修改至少要覆盖：run-root/config composition、legacy/canonical resume path、
checkpoint round-trip、多个 worker 的 completed/success 聚合，以及“chunks 不推进 real/eval
主进度”的测试。完成前还要执行相关单元测试、Ruff、格式检查、Shell 语法检查和
`git diff --check`。不得仅凭进程成功启动就宣称这些约束已经满足。
