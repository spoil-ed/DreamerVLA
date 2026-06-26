# 2026-06-25 Cotrain Pipeline 决议与执行计划

本文档总结 online cotrain pipeline 的抽象决议、争议处理和下一步代码执行口径。
它的目标不是复述具体代码，而是固定“训练 loop 应该是什么”，然后让代码实现按这个语义验收。

适用范围：

- cold-start collect -> replay seed -> WM warmup -> classifier warmup -> online cotrain。
- 默认 VLA 配置可以是 OpenVLA-OFT discrete/action-hidden 路线，但不能写死 OpenVLA-OFT。
- VLA、world model、classifier、dataset、runner 必须继续通过 Hydra target、registry、protocol 或窄接口解耦。

## 0. 总体结论

完整 pipeline 应按以下闭环理解：

```text
真实环境采集完整 episode
-> 完整 episode 进入 real replay
-> WM 从 real replay 学真实状态/hidden 转移
-> classifier 从 real replay 学成功/失败判别
-> actor 从 replay 起点出发，在 WM 内生成 imagined rollouts
-> classifier 对 imagined rollouts 打分
-> PPO/GRPO 根据同一起点多条 imagined rollout 的 outcome 差异更新 actor
-> 更新后的 actor 进入后续真实 online rollout
-> 新真实 episode 继续追加 real replay
-> 循环
```

最重要的约束：

- 真实 success rate 只能来自 completed real episode。
- imagined score 只能作为 PPO 学习信号和诊断，不能当真实成功率。
- 真实 rollout、WM training、PPO imagination 都必须按 action chunk 语义对齐。
- full run 需要看到 warmup、actor update、updated actor rollout 三者闭合，smoke 只能证明链路不崩。

## 1. VLA 配置不是硬编码模型

OpenVLA-OFT 应写成“默认 VLA 配置”或“默认 OpenVLA-OFT discrete/action-hidden 路线”，不能写成 pipeline 的硬依赖。

原因：

- 当前配置已经通过 `task.openvla_oft.*`、Hydra `_target_`、组件 config 等方式选择具体实现。
- 未来 RynnVLA、其他 VLA encoder、其他 actor head 应能复用同一 runner/learner 语义。
- AGENTS.md 要求按 role 命名和解耦，不能让类名、runner 逻辑或文档把某个 checkpoint/dataset 当作固定实现。

执行要求：

- 文档里使用“默认 VLA 配置”。
- 代码里通过 config 选择 encoder、policy、WM、classifier。
- 不在训练 loop 中用 OpenVLA-OFT/RynnVLA 字符串决定训练语义；仅允许在外部边界适配器中处理具体模型 I/O。

## 2. Replay 分层

Replay 分两层语义，但默认只长期保存 real replay。

real replay：

```text
内容：
  offline cold-start complete episodes
  online real complete episodes

用途：
  WM warmup / cotrain
  classifier warmup / cotrain
  PPO 起点采样

生命周期：
  长期保存或长期驻留
```

imagined rollout buffer：

```text
内容：
  actor 在 WM 内生成的 imagined trajectories
  classifier scores / outcome / returns / advantages

用途：
  单次 PPO/GRPO actor update

生命周期：
  PPO step 内临时生成、临时消费
  默认不写入长期 replay
```

暂不把 imagined trajectory 做成长期 Ray replay layer。

理由：

- 调试会更清楚，但会引入状态同步、显存/内存占用、policy staleness 和 replay 污染问题。
- 当前目标是先让真实数据 replay 与 imagined PPO update 的边界清楚。
- 只保留必要诊断摘要即可，例如 score mean/std、mixed group 比例、skipped zero-variance groups。

Ray 抽象：

- Ray 下可以用 ReplayWorker 包装同一个 OnlineReplay。
- no-Ray 和 Ray 应共享 real replay 语义。
- imagined rollout 如果以后要 Ray 化，应作为短生命周期 PPO worker 内部对象，而不是混进 real replay。

## 3. Replay Ready 语义

“一条 episode 就够”只适用于 smoke，不适用于 full training。

smoke ready：

```text
每个 active task >= 1 complete episode
总 transition 数 >= sequence_length
可以跑通 replay.sample()
可以跑通 WM/classifier/RL step
```

full training ready：

```text
每个 active task >= N complete episodes
总 transitions 足够
sampleable WM windows 足够覆盖至少一个 replay epoch
classifier candidate windows 同时有正负证据
replay 没有被 capacity 截断到只剩很少窗口
```

当前推荐：

- `min_episodes_per_task=1` 只能视为最低启动门槛。
- full run 应额外检查 sampleable windows、classifier positive/negative evidence、每任务 coverage。
- N 的具体值根据实际 cold-start 数据和任务难度设置，不在训练 loop 里硬编码。

## 4. 三池采样

cotrain 开始后仍需要保留 cold-start 数据，但采样应偏向新 online 数据。

采用三池采样：

```text
online_recent:
  最近 N 条 online episodes。
  最高优先级，特别是 latest online episode 应优先进入 learner update。

online_replay:
  全部 online episodes。
  用于保持 online 分布多样性。

coldstart_anchor:
  offline cold-start episodes。
  用作稳定 anchor，避免模型完全忘掉初始可用轨迹。
```

默认采样策略：

```text
online_recent    0.5
online_replay    0.3
coldstart_anchor 0.2
latest_online_required = true
```

注意：

- 新数据优先不等于丢掉旧数据。
- cold-start anchor 不应占太高比例，否则 online adaptation 变慢。
- 如果某个池为空，采样器应自动 fallback 到可用池。

## 5. Warmup 的时间单位

Warmup 不应只用 fixed steps 表达。更合理的单位是 `replay_epoch`。

定义：

```text
WM replay_epoch:
  对当前 sampleable WM sequence windows 的一次覆盖。

classifier replay_epoch:
  对当前 classifier candidate windows 的一次覆盖。
```

执行建议：

- 支持 `warmup_replay_epochs`。
- 支持 `warmup_replay_max_steps` 作为工程上限，避免一次 full coverage 过长。
- 继续保留 fixed step 配置作为兼容路径。

当前经验参考：

```text
500 episodes / 100633 transitions
WM 2000 steps
global batch 约 24
粗略只覆盖约半个 sampleable-window epoch
```

因此 full warmup 不应默认只相信 2000 fixed steps。应根据实际 sampleable window count 决定 2-3 个 replay epoch 或相应上限。

## 6. WM 与 Classifier Warmup 排序

不强制 WM 和 classifier 使用同一个 tensor batch。

原因：

- WM batch 需要连续 sequence windows。
- classifier batch 需要成功/失败 candidate windows。
- 两者来自同一 replay pool，但天然不是完全同构样本。

推荐方案：

```text
replay epoch 内交替：
  WM batch update
  classifier window batch update
  WM batch update
  classifier window batch update
  ...
```

优点：

- 保持 replay 分布同步。
- 日志能同时观察 WM 与 classifier 的变化。
- 实现复杂度低于“同一个 cursor 同一个 tensor batch”。

暂不采用：

- 先完整 WM warmup 再完整 classifier warmup 作为唯一 full 路径。
- 强行把 WM sequence batch 和 classifier window batch 合成同一个 batch。

## 7. PPO Imagined Rollout 条数

不能无限采到成功为止。

推荐 bounded adaptive group：

```text
每个起点最少 K_min = 4 条 imagined rollouts
如果 classifier score / outcome 没有方差，则继续补采
最多 K_max = 16
若 K_max 仍无方差，则跳过该起点 actor update
```

需要记录：

```text
rl/returns_mean
rl/returns_std
rl/policy_grad_norm
rl/skipped_zero_variance_groups
LUMOS/score_mean
LUMOS/score_std
LUMOS/group_var_keep_frac
LUMOS/num_mixed_groups
```

“选择性过滤失败，改变真实分布”的含义：

- 如果一直采到出现成功才更新，训练样本会偏向“有成功信号的 imagined group”。
- 大量全失败或全成功 group 被隐式丢弃，会让 PPO 看到的分布不同于 actor 当前真实分布。
- 这可能制造表面学习信号，但 actor 学到的是被筛选后的分布。

所以：

- 可以跳过无方差 group。
- 不能无限补采直到成功。
- 必须记录 skipped/mixed 比例，判断 classifier 和 WM 是否提供了健康信号。

## 8. Success Rate 与 Eval

指标必须少，但语义要准。

最少保留：

```text
rollout/success_rate:
  completed real episodes 的累计成功率。

rollout/recent_success_rate:
  最近 N 个 completed real episodes 的成功率。

rollout/success_rate_valid:
  至少完成一个真实 episode 后为 1。

rollout/episodes:
  completed real episodes 数。
```

用户口径中的“最近几次平均成功率”对应 `rollout/recent_success_rate`，窗口单位是 episode，不是 step。

提升判断：

- 在线趋势主要看 recent window success rate。
- 最终声明提升应看 periodic real eval 或固定预算 real rollout。
- cumulative success rate 只能作为辅助，因为早期失败会长期稀释后续提升。

Eval 触发建议：

```text
每采集一定数量 online completed episodes 后 eval 一次
或每若干 learner_update/global_update_step eval 一次
full run 至少要有 update 后的真实 rollout/eval
```

## 9. RLinf 时间单位对齐

Online cotrain 不称为 epoch。

推荐命名：

```text
env_step:
  真实环境 low-level action 执行一步。

episode:
  从 reset 到 success/done/horizon/truncation 的完整真实轨迹。

rollout:
  一段真实或 imagined 轨迹；必须说明 real 还是 imagined。

learner_update:
  WM/classifier/actor optimizer update。

global_update_step:
  learner 侧全局更新计数。

replay_epoch:
  warmup 阶段对 replay sampleable windows 的覆盖 pass。

eval_window:
  固定 episode 数或固定评估预算。
```

和 RLinf 对齐的核心不是照搬 epoch，而是：

- rollout 与 learner 可以 overlap。
- worker/component placement 明确。
- metrics namespace 清楚。
- checkpoint / resume / logger 统一。
- online loop 用 env_step、episode、learner_update 表达。

## 10. Task Conditioning

当前 hidden 很可能已经包含任务语言信息。

依据：

- online OFT extractor 由 `task_description` 构造 prompt。
- VLA forward 同时使用图像、语言 token、attention mask。
- offline sidecar 通常保存来自 task prompt 条件下的 hidden。

但多任务 full cotrain 仍建议加入显式 task conditioning。

理由：

- replay 已经有 `task_id`。
- 多任务下相似视觉状态可能对应不同目标。
- classifier 没有 task condition 时可能混淆不同任务的成功状态。
- 显式 task conditioning 便于做消融和回退。

推荐实现：

```text
task_conditioning.enabled:
  默认 single-task false。
  multi-task full run 可开启。

task_conditioning.num_tasks:
  由 task suite / env.task_ids 给出。

task_conditioning.embedding_dim:
  Hydra 配置，不在代码中硬编码。
```

验收要求：

- replay batch 带 `task_ids`。
- WM forward 可接收 `task_ids`，或在开启 task_conditioning 时明确报错。
- classifier forward 可接收 `task_ids`，或在开启 task_conditioning 时明确报错。
- 默认关闭时不影响现有单任务路径。

## 11. 真实 Rollout 必须执行完整 Action Chunk

这是最高优先级风险之一。

正确语义：

```text
actor 输出 K-step action chunk
真实 env 按顺序执行 chunk 内所有 low-level actions
chunk 执行期间不重新采样 actor
chunk 用完后再根据新 observation/hidden 重新规划
episode 结束或 reset 时丢弃未执行完的 pending chunk
```

原因：

- PPO imagination 是 chunk 级。
- WM 是 chunk-aware。
- 如果真实 rollout 每步只执行 chunk[0]，真实数据分布会和 imagined rollout 不一致。
- 这会造成 actor 在 WM 里学到的行为无法正确进入真实环境。

动作 scale 合约：

learned actor path：

```text
actor 输出 normalized action
env(action_input=normalized) 将其映射成 raw LIBERO action
env info['wm_action'] 记录 raw executed action
WM/replay 使用 raw wm_action
```

OFT fixed-base path：

```text
OFT base 输出 raw-ish action chunk
rollout worker 做 gripper postprocess
env(action_input=raw) 执行
WM/replay 使用 raw wm_action
```

不能把两条路径混用。

执行要求：

- no-Ray learned actor rollout 必须维护 per-env pending action queue。
- Ray generic learned actor InferenceWorker 也必须维护 per-env pending action queue。
- Ray OFT fixed-base RolloutInferenceWorker 的 `action_steps` 应等于 chunk_size，不能默认只执行第一步。
- observe_next 的 previous action 应使用 env 返回的 raw `wm_action`，不是 normalized actor action。

## 12. Metadata

不需要新增 step-level metadata attrs。

推荐分层：

HDF5 per-demo attrs：

```text
suite
task_name
task_id
episode_id
global_episode_index
policy_name
policy_ckpt
policy_version
success
success_step
horizon
timeout
chunk_size
action_scale
seed
render_backend
hidden_key
hidden_dim
token_count
token_dim
```

HDF5 datasets：

```text
actions
rewards
dones
sparse_rewards
obs / image
states / proprio
obs_embedding
```

collection_manifest.json：

```text
suite
target episodes
collected counts
policy checkpoint
hidden schema
backend
shard list
created time
resolved config snapshot
resume status
```

原则：

- attrs 存 episode-level 标量/字符串。
- 数组型逐步信息作为 dataset。
- 不把每步 metadata 塞进 attrs。

## 13. WM 是 Chunk-Aware End-to-End

当前主路线保持 chunk-aware WM，不退回普通 step WM。

更准确的描述：

```text
warmup 输入:
  env-step sequence window

外部训练目标:
  chunk-aware end-to-end prediction

内部实现:
  可以 step-recursive，也可以并行预测 chunk
```

用户给出的公式可以理解为一种 autoregressive/chunk 内递推形式：

```text
pred_1 = WM(real_1, real_2, real_3, action_1)
pred_2 = WM(real_2, real_3, pred_1, action_2)
pred_3 = WM(real_3, pred_1, pred_2, action_3)
pred_4 = WM(pred_1, pred_2, pred_3, action_4)
```

这个思想是合理的：真实历史提供 burn-in，后续逐渐用预测 hidden 继续 rollout。

但外部接口仍应保持：

```text
WM 接收完整 action sequence/chunk
WM 并行或半并行计算训练 loss
WM rollout 输出 chunk-level imagined hidden sequence
```

不建议把训练接口退回“单 step 调一次 Python loop”的低效形式。

## 14. Classifier Chunk Pooling 的边界

“chunk 用了 pool”只发生在 classifier window handling，不代表 WM chunk 被 pool 掉。

当前合理语义：

```text
replay 中有 env-step obs_embedding
classifier window 配置为 W 个 chunk
实际先取 W * chunk_size 个 env-step hidden
再按 chunk_pool=last|first|mean 聚合成 W 个 classifier frames
classifier 对 W 个 chunk frames 打分
```

这只是 classifier 的输入压缩策略。

WM 主路线仍是：

- chunk-aware。
- action sequence/chunk 作为条件。
- hidden rollout 保持 chunk 语义。

如果后续要避免 classifier pooling，可以让 classifier 直接吃完整 `W * chunk_size` env-step hidden 或 token sequence，但这会提高显存和计算量，应作为单独实验。

## 15. Debug 与 Full 的区别

debug 可以降低预算，但不能改变算法语义。

可以改：

```text
总 env steps
warmup replay epochs / max steps
batch size
num_envs / worker 数
eval budget
checkpoint interval
```

不应改：

```text
action scale
chunk execution semantics
rollout policy source
reward/outcome 定义
real vs imagined success rate 语义
task conditioning 开关的默认解释
episode completion condition
```

full run 必须覆盖：

```text
cold-start collection
WM warmup
classifier warmup
online real rollout
WM/classifier learner update
PPO/GRPO actor update
updated actor 后续 real rollout 或 real eval
checkpoint save/resume/load
```

## 16. 指标最小集合

为了避免指标过多，默认重点保留：

真实 rollout：

```text
rollout/episodes
rollout/env_steps
rollout/success_rate
rollout/success_rate_valid
rollout/recent_success_rate
rollout/recent_success_rate_valid
```

WM：

```text
wm/loss
wm/hidden_rec_loss
wm/hidden_cosine_loss
wm/full_hidden_rec_loss
wm/full_hidden_cosine_loss
```

Classifier：

```text
cls/loss
cls/f1
cls/pos_frac
cls/grad_norm
```

Actor / PPO：

```text
rl/actor_loss
rl/returns_mean
rl/returns_std
rl/policy_grad_norm
rl/ppo_step_applied
rl/skipped_zero_variance_groups
LUMOS/score_mean
LUMOS/score_std
LUMOS/group_var_keep_frac
```

解释：

- `score_mean` 是 classifier 对 imagined rollout 的成功分数均值，不是真实成功率。
- `rollout/success_rate` 才是真实 completed episode 统计。
- `returns_std` 或 `score_std` 长期为 0 时，PPO 没有有效优势信号。

## 17. 执行优先级

P0：

```text
修真实 rollout 只执行 chunk 第一个 action 的问题。
保证真实 rollout、WM replay action、PPO imagination 的 chunk/action scale 一致。
每个 completed episode 后输出当前 success、累计 success rate、recent success rate。
```

P1：

```text
warmup 支持 replay_epoch 语义，并保留 max step cap。
real replay 使用三池采样，latest online episode 优先。
full ready 条件补充 classifier 正负证据与 sampleable window coverage。
WM/classifier warmup 改成 replay 内交替 datastream。
```

P1 multi-task：

```text
添加可回退 task_conditioning.enabled。
replay batch 传 task_ids。
WM/classifier 声明 supports_task_conditioning 后才允许开启。
```

P2：

```text
adaptive imagined rollout group: K_min=4, K_max=16。
补充 episode-level metadata。
periodic real eval 接入 online cotrain loop。
Ray OFT fixed-base rollout 与 Ray learned-actor rollout 明确分模式。
```

## 18. Go / No-Go

不启动 full training 的条件：

```text
真实 rollout action chunk 未对齐
action scale 合约未验证
classifier 没有正负判别能力
PPO imagined outcome 长期无方差
真实 success rate 统计不是 completed episode denominator
updated actor 没有进入后续真实 rollout/eval
```

可以启动 full training 的最低条件：

```text
Part A 逻辑测试全绿
A3 classifier f1 >= 0.6
A4 action scale 静态自洽
full warmup 覆盖足够 replay windows
online rollout 每个 completed episode 输出 cumulative + recent success rate
PPO 至少出现非零 actor_loss / policy_grad_norm / returns_std
```

最终成功判据：

```text
先证明信号活：
  classifier 可判别
  imagined returns 有方差
  PPO actor grad 非零

再证明真实改善：
  updated actor 后续真实 rollout/eval 的 recent-window success rate 高于 base/early window
```
