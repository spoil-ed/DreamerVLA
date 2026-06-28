# DreamerVLA Spec

本目录是 DreamerVLA 当前方案的 architecture/spec 根目录。目标是：只阅读
`spec/`，就能理解当前主线为什么这样设计、训练流程如何运行、各个 group 的边界是什么、
哪些内容已经实现、哪些内容仍需要验证或继续补充。

这里不是代码 API 索引，也不是历史日志。代码路径、类名、命令和配置键只用于辅助定位；
正文应优先解释方案本身。

## Source Of Truth

`99_manual_notes.md` 是用户第一性指导，优先级最高。除非用户明确要求，AI agent 不得移动、
压缩、重写或删除其中的用户手写内容。

整理后的主文档用于把第一性指导转成稳定、可读、可执行的方案说明：

- `00_overview.md`：说明主线架构和核心思想。
- `01_complete_loop.md`：说明 collect、warmup、cotrain、eval 的端到端流程。
- `02_ray.md`：说明 Ray group、worker、channel、placement 和同步边界。
- `98_prompt.md`：记录任务提示或迁移背景，不作为当前方案的最高优先级来源。
- `99_manual_notes.md`：用户第一性指导和目标方向，优先级高于整理后的文档。
- `superpowers/plans/`：执行计划和实现过程记录。可以作为历史参考，但不要把它当作当前
  architecture source of truth。

如果文档之间出现冲突，处理顺序是：

1. 先以 `99_manual_notes.md` 的第一性判断为准。
2. 再检查当前代码和测试是否已经落地了新的事实。
3. 把差异写成 `Current vs Target` 或 `Open Questions`，不要直接混成一句含糊描述。
4. 只有确认后，才更新主文档中的目标方案或当前事实。

## Writing Style

本目录的文档以自然语言描述为主。每个章节应先让人理解“为什么这样设计、谁负责什么、数据如何流动、
哪些边界不能破坏”，再用 API contract 固定实现边界。

推荐结构：

````markdown
## Component Or Flow

先用自然语言说明这个组件或流程的意义。这里要解释设计意图、职责边界、数据流、
同步关系和重要取舍。不要一开始就堆类名、函数名或配置键。

### API Contract

```text
Component.method(
  arg: Type,        # 用注释解释这个参数的语义，不解释源码实现
) -> ReturnType     # 用注释说明返回值在流程中的作用
```

约束：

- 写清楚必须满足的 shape、版本、同步顺序和禁止事项。
- 只列真正影响实现边界的 API，不把源码所有方法照抄进来。
- 如果某个字段只是当前实现细节，不要把它写成长期架构约束。
````

一个合格章节通常应该同时回答：

- 这个组件为什么存在？
- 它负责什么，不负责什么？
- 它从哪里拿数据，向哪里发送数据？
- 它和其它组件如何同步版本或权重？
- 失败时最可能破坏哪个契约？
- 当前代码是否已经实现？如果没有，差距在哪里？

## API Contract With Comments

API contract 应该使用“带注释的伪接口”。注释是语义说明，不是逐行代码解释。

示例：

```text
ActorGroup.sync_model_to_rollout(
  key: "policy",        # 这里只同步 VLA policy，不同步 world model/classifier
  version: int,         # 通常等于当前 global_step，用于 rollout policy version
) -> dict[str, float]   # 至少应记录 sync/policy_version

ActorGroup.load_trajectory_shards(
  shards: list[TrajectoryShard],  # EnvGroup 直接发来的 trajectory，不是 replay sample
) -> None
```

消息结构也应写成带注释的 contract：

```text
TrajectoryShard:
  actions: [T, B, chunk, action_dim]
    # EnvGroup 实际执行过的 action chunk，也是 ActorGroup 重新算 logprob 的目标动作

  prev_logprobs: [T, B]
    # RolloutGroup 采样该 action 时的 old logprob

  forward_inputs: dict[str, Tensor]
    # ActorGroup 重新计算 new_logprobs 所需的完整输入
    # 如果 VLA encoder 参与训练，这里不能只保存 detached embedding

  versions.policy: [T, B]
    # 产生该 trajectory 的 rollout policy version
```

全局流程可以写成带注释的状态机：

```text
1. ActorGroup.set_global_step(step)
   # Actor 进入当前训练版本

2. ActorGroup.sync_model_to_rollout("policy", step)
   # Actor -> Rollout；Actor 内部 FSDP/NCCL 同步不走这里

3. EnvGroup.interact(env_channel, rollout_channel, actor_channel)
   # EnvWorker 推进真实环境或 WMEnv，并组装 trajectory

4. RolloutGroup.generate(env_channel, rollout_channel)
   # RolloutWorker no-grad 推理，返回 action/logprob/forward_inputs

5. ActorGroup.run_training()
   # PPO clipped loss + backward + optimizer step

6. LearnerGroup.update("cotrain", num_steps)
   # 只更新 world model 和 classifier，不训练 VLA
```

## Status Labels

每个文档或关键章节应显式标注状态，避免目标方案、当前实现和历史记录混在一起。

推荐使用：

- `状态：current`：描述当前代码已经实现并有测试或运行记录支撑的事实。
- `状态：target`：描述用户确认的目标方案，但不保证当前代码已经完全实现。
- `状态：mixed`：同一章节同时包含当前事实和目标差距，必须写 `Current vs Target`。
- `状态：historical`：历史提示、旧计划、迁移记录，只能作为背景参考。

如果写到了代码事实，尽量补一小段定位信息：

```markdown
代码定位：
- runner: `dreamervla/runners/manual_cotrain_ray_runner.py`
- placement: `dreamervla/workers/cotrain/placement.py`
- messages: `dreamervla/workers/cotrain/messages.py`
```

代码定位只帮助读者继续查证，不应替代自然语言解释。

## Important Boundaries

写 `spec/` 时需要特别保护这些边界：

- ActorGroup 和 RolloutGroup 是不同角色。RolloutGroup 负责 no-grad 行为策略推理；
  ActorGroup 负责训练中的 VLA policy 和 PPO update。
- LearnerGroup 不训练 VLA。它只训练 world model 和 classifier/reward model。
- WMEnvWorker 使用 LearnerGroup 同步来的 world model 和 classifier/reward model。
- Actor PPO 主线直接消费 EnvGroup 发来的 trajectory shard，不把 replay sample 当作
  ActorGroup 的 PPO batch 来源。
- ActorGroup 内部 FSDP 同步由 FSDP/NCCL 管理；Actor -> Rollout 是另一层权重同步。
- `forward_inputs` 是 ActorGroup 重新计算 logprob 的关键契约，不能随意弱化。
- 真实 LIBERO eval 是最终效果判断；WMEnv reward 只是训练信号的一部分。
- Hydra 配置是运行 source of truth。文档可以解释配置语义，但不要鼓励在代码里硬编码行为。

## Constraints To Fill Later

下面是后续需要继续补齐的约束区。填写时仍然遵循“自然语言先行 + API contract 加注释”的格式。

### Resource Profiles

待填写：

- 0 GPU、1 GPU、2-5 GPU、6+ GPU 的推荐启动形态。
- EGL 和 osmesa 的选择规则。
- 每张卡上 RealEnv、WMEnv、Rollout、Actor、Learner 的默认放置策略。
- 哪些 profile 是 tiny smoke，哪些 profile 是真实训练。

### Data And Artifact Contracts

待填写：

- collected rollout reward shard 和 hidden sidecar 的目录、字段和生命周期。
- replay 中哪些数据用于 LearnerGroup，哪些数据不得替代 ActorGroup trajectory。
- warmup checkpoint、manual cotrain checkpoint、eval artifact 的保存位置和恢复规则。

### Verification Matrix

待填写：

- 哪些 unit tests 证明消息、placement、config 和 runner 边界。
- 哪些 e2e smoke 证明 tiny manual cotrain 可以完成一个 global_step。
- 哪条命令用于验证真实 OpenVLA-OFT + LIBERO + Ray async cotrain 长时间启动稳定。
- 每种验证失败时应该优先检查的日志和配置。

### Open Decisions

待填写：

- 是否长期保留旧 Ray async route，还是只作为 explicit legacy route。
- value-based PPO/GAE 是否进入主线；如果进入，final bootstrap value 的契约是什么。
- VLA encoder 是否参与 ActorGroup 训练；如果参与，`forward_inputs` 需要保存哪些非 detached 输入。
- WMEnv reward/classifier 的版本同步频率和 stale policy 容忍范围。

## Agent Rules

AI agent 在本目录写文档时必须遵守：

- 不要把未经确认的推断写成当前事实。
- 不要删除或压缩用户第一性指导。
- 不要把执行计划文件当作最终 architecture 文档。
- 不要只列代码路径；必须先解释方案。
- 不要只写自然语言而省略关键 API contract；涉及实现边界时必须补带注释的 contract。
- 新增或修改主线文档时，必须显式说明该段是 `current`、`target`、`mixed` 还是 `historical`。
- 如果文档描述了可运行流程，应同时写出最小验证方式或说明验证仍缺失。

