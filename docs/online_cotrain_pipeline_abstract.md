# Online Cotrain Abstract Pipeline

本文档定义 cold-start collect -> warmup -> online cotrain 的抽象训练流程。它描述数据流、
时间单位、成功率语义和停止条件；具体类、函数和配置事实由代码审计确认。

OpenVLA-OFT 在本教程路线中是默认离散 VLA 配置，不是硬编码架构依赖。VLA、world model、
classifier、reward 与 runner 必须继续通过 Hydra target、registry、protocol 或窄接口选择，
以便替换为 RynnVLA、其他 WM 或其他 classifier。

## Settled Decisions

1. Real replay 只保存完整真实 episode：cold-start episodes 与 online episodes。
2. Imagined rollouts 是 learner 临时内存或 PPO/outcome tensors，不进入真实 replay。
3. Cotrain 开始后偏向最新 online 数据，但 cold-start 数据保留为 anchor。
4. Replay sampling 使用三池视图：`online_recent`、`online_replay`、`coldstart_anchor`。
5. Latest online episode 应在 learner update 中高优先级采样，避免在线数据长时间不可见。
6. Real rollout 必须按顺序执行完整 VLA action chunk。
7. Online episode metadata 只要求 episode-level；不新增 step-level metadata 要求。
8. Warmup 使用 replay coverage 语义：一次覆盖当前 sampleable replay windows 称为
   `replay_epoch`。Online cotrain loop 不称为 epoch。
9. RLinf-style 在线时间单位是 rollout 的 `env_step` / `episode`，以及 learner/PPO 的
   `learner_update` / `global_update_step`。
10. Multitask cotrain 增加可选显式 task conditioning，默认关闭并标记可回退；即使 hidden
    已包含语言/任务信息，显式 conditioning 仍用于多任务消融和稳健性。
11. Debug/smoke 只能证明链路执行；full/release run 才能判断训练质量。
12. WM 外部训练目标保持 chunk-aware end-to-end；内部预测可以 step-recursive。

## Time Units

1. `env_step`：真实环境执行一个 low-level action 后前进一步。
2. `episode`：从 reset 到 env success/done 或 horizon/truncation 的完整真实轨迹。
3. `replay sample`：从 replay 采样的真实数据窗口。
4. `replay_epoch`：warmup 中对 sampleable replay windows 的一次覆盖 pass。
5. `learner_update` / `global_update_step`：WM、classifier 或 actor optimizer 更新。
6. `imagined rollout`：world model 内由 actor 生成的临时未来轨迹。
7. `eval window`：固定数量真实 episodes 或固定评估预算。

## Phase 0: Route Selection And Run Roots

Start condition:
Hydra 已选择 experiment、task、VLA、WM、classifier、reward、runner、logger、checkpoint
与 rollout backend。collect backend 可以是 no-Ray 或 Ray；online render backend 可以是
osmesa 或 egl。

End condition:
配置通过早期 validation，run root 确定，输出位置限定在 `${training.out_dir}` 及其阶段子目录。

Stop / rollback condition:
组件缺少 `_target_`、维度无法从 VLA/task/sidecar/config 对齐、checkpoint 路径不明确、
或 Ray 被隐式设为默认拓扑时停止。

Time unit:
配置解析阶段，无训练时间单位。

## Phase 1: Cold-Start Real Episode Collection

Start condition:
初始 VLA policy 可加载，真实 LIBERO task/env 可 reset，collect target episodes 已配置。

End condition:
达到配置的 cold-start completed episode 数，或已有可恢复 shard 满足目标。

Stop / rollback condition:
env 无法 reset/render、policy load 失败、action scale 明显不一致、或 episode 无法按
success/done/horizon 完整结束。

Time unit:
`env_step` 与 `episode`。

## Phase 2: Replay Initialization From Complete Episodes

Start condition:
cold-start collect 输出完整真实 episode。磁盘上通常包含原始 reward/rollout HDF5 与 hidden
sidecar HDF5；原始 HDF5 是真实轨迹来源，hidden sidecar 用于加速 warmup 和 learner 读取。

End condition:
完整 episodes 已载入 replay，source 标记为 `coldstart`，task/success metadata 可用于采样诊断。

Stop / rollback condition:
episode 不完整、hidden/action/reward 时间对不齐、sidecar 维度不匹配、或 replay 容量截断导致
full warmup 覆盖假象。

Time unit:
`episode` 与 `replay sample`。

## Phase 3: World Model Warmup

Start condition:
Replay 已由 cold-start complete episodes 初始化，并满足配置的 sampleable windows、task coverage
和 transition minima。

End condition:
达到固定 `wm_warmup_steps`，或按 `warmup_replay_epochs` 完成配置数量的 replay coverage；
若设置 `warmup_replay_max_steps`，coverage-derived learner updates 会被该上限截断。

Stop / rollback condition:
WM loss NaN/inf、hidden reconstruction 长期无效、chunk/action 语义不对齐、或 sampleable replay
不足以支撑 full warmup。

Time unit:
`replay_epoch` 和 `learner_update`。

## Phase 4: Classifier Warmup

Start condition:
Replay 中存在可采样 classifier windows。Full cotrain 下应有正负证据；smoke 可以只验证链路。

End condition:
达到固定 classifier warmup steps，或按 `warmup_replay_epochs` 完成配置覆盖；若设置
`warmup_replay_max_steps`，coverage-derived learner updates 会被该上限截断。WM 和 classifier
使用同一 replay pool 的两条 datastream 交替更新，不要求同一 tensor batch。

Stop / rollback condition:
只有单一类别、标签与窗口错位、F1 长期接近随机或全零、或 classifier 过早饱和。

Time unit:
`replay_epoch` 和 `learner_update`。

## Phase 5: Online Real Rollout

Start condition:
Warmup 达到配置门槛，online env 可 reset，当前 actor 或同步 actor copy 可用于真实 rollout。

End condition:
执行配置的 online env-step budget，或产生一个或多个 completed real episodes。

Stop / rollback condition:
动作执行异常、render backend 反复失败、episode 无法完成、或 total env-step budget 已达到。

Time unit:
`env_step` 和 `episode`。

## Phase 6: Episode-End Replay Append

Start condition:
某个 online rollout slot 到达 env success/done 或 horizon/truncation。

End condition:
完整 episode 追加到 replay，source 标记为 `online`，真实成功率统计更新。

Stop / rollback condition:
partial episode 不写入 replay；reset 时必须丢弃未执行完的 pending action chunk。

Time unit:
`episode`。

## Phase 7: Learner Updates From Replay

Start condition:
Replay ready 满足配置 minima。一个 episode 可让 replay 非空，但 full cotrain ready 还依赖
transition 数、task coverage、每任务 episode 数和 classifier 正负证据。

End condition:
完成配置的 learner updates 或达到 `max_train_updates`。

Stop / rollback condition:
Replay 未 ready、DDP ranks 不同步、梯度 NaN、或训练信号诊断显示 WM/classifier 不可用。

Time unit:
`learner_update` / `global_update_step`。

## Phase 8: Imagined Rollout In The World Model

Start condition:
Learner 从 replay 采样真实起点，WM 可从该 state/action boundary 预测未来 hidden state。

End condition:
每个起点生成至少 `K_min=4` 条 imagined rollouts；若 score 方差出现即可使用当前组，最多到
`K_max=16`。

Stop / rollback condition:
到 `K_max` 仍无 classifier score 方差时跳过该起点 actor update，并记录 skipped group。
Imagined rollout 不写入真实 replay。

Time unit:
`imagined rollout`。

## Phase 9: Classifier Scoring Of Imagined Trajectories

Start condition:
Imagined hidden/action trajectory 已生成，classifier 可接受对应 window 表示。

End condition:
每条 imagined trajectory 得到 success-style score/probability/outcome。Sparse outcome 规则是：
若任一 scored window 成功，可把该 trajectory 视作成功 outcome；概率和 score 诊断继续保留。

Stop / rollback condition:
Classifier 输入窗口语义不清、score 全饱和、或把 imagined score 误报为真实 success rate。

Time unit:
`imagined rollout`。

## Phase 10: PPO/GRPO Actor Update

Start condition:
同一起点的 imagined outcome 存在足够方差。

End condition:
Actor 通过 PPO/GRPO 完成一次或一组 `learner_update`，并记录 actor loss、returns variance、
policy grad norm 与 skipped zero-variance groups。

Stop / rollback condition:
Outcome 无方差、policy grad norm 长期为零、或 actor update 被误写成只模仿成功真实轨迹。

Time unit:
`learner_update` / `global_update_step`。

## Phase 11: Next Online Rollout Uses Updated Actor

Start condition:
Actor update 完成，或 Ray inference worker 收到同步后的 actor weights。

End condition:
后续真实 online rollout 自然使用更新后的 actor 或其同步 copy。

Stop / rollback condition:
Actor 只在 learner 内更新但从未进入真实 rollout/eval 时，不能声称真实成功率提升。

Time unit:
`env_step` 与 `episode`。

## Phase 12: Evaluation And Stop Decision

Start condition:
达到配置的 eval interval、training budget、early-stop condition，或需要证明真实效果。

End condition:
得到 completed real episodes 的 cumulative success rate 与 recent-window success rate，
并保存 checkpoint/metrics。

Stop / rollback condition:
真实 success statistics 无效、classifier F1 退化、imagined returns 无方差、actor grad 为零、
OOM 或 render failure 重复出现。

Time unit:
`eval window`、`episode`、`global_update_step`。

## Metrics Semantics

1. `rollout/success_rate`：completed real episodes 的累计成功率。
2. `rollout/success_rate_valid`：至少完成一个真实 episode 后为 1.0。
3. `rollout/recent_success_rate`：最近配置窗口内 completed real episodes 的成功率。
4. `rollout/recent_success_rate_valid`：recent window 至少有一个 completed episode 后为 1.0。
5. `rollout/episodes`：累计 completed real episodes。
6. `rollout/env_steps`：累计真实环境步数。
7. `rl/returns_mean` 与 `rl/returns_std`：imagined actor signal 诊断，不是真实成功率。
8. `rl/skipped_zero_variance_groups`：actor signal 诊断，不是真实成功率。

## Multitask Conditioning

当前 hidden state 已通过 VLA prompt 路径携带语言或 task prompt 信息。online OFT
extractor 由 `task_description` 构造 prompt，并把 `input_ids` / `attention_mask`
连同图像输入同一次 VLA forward；offline OFT/RynnVLA sidecar 也记录 task prompt 或
actor token 序列。因此，多 task cotrain 的显式 task conditioning 是可选增强，用于
消融和稳健性，而不是补齐唯一 task 信息来源。

该功能默认关闭；启用时 WM/classifier 从 replay batch 接收 task id。若启用但所选实现
不支持，应在 config validation 阶段失败。该功能应在配置和文档中标明可回退。

## Code-Confirmed Boundaries

1. WM state boundary 由 Hydra/sidecar 的 `obs_hidden_source` 决定，而不是 runner
   内的模型名称分支。`action_query` 表示 action-query/action-hidden 边界；
   `input_token_embedding` 表示 projected input-token/backbone-token 边界。
2. Replay batch 的 `obs_embedding` 沿用上述 sidecar/online extractor 边界；downstream
   WM/classifier/actor 只按配置维度消费该隐藏状态，不重新推断外部 hidden 语义。
3. 当前 classifier window 的原始单位是 env-step frames。`OnlineReplay.sample_classifier_windows`
   用 `window * chunk_size` 个 env-step `obs_embedding` 构造窗口；当 `chunk_size > 1`
   时，按 `chunk_pool=last|first|mean` 聚合成 W 个 classifier frames。
4. `LatentSuccessClassifier.predict_success` 对 chunk granularity 使用相同的
   `chunk_size/chunk_pool` 聚合规则；这属于 classifier-window handling，不改变 WM
   chunk-aware external training target。

## Remaining Runtime Verification

1. 默认 torch checkpoint 路径已经支持 save/resume/load；HF 组件边界仍需按实际选择的组件
   继续核验。
2. no-Ray online cotrain 是否需要异步实现另行决定；现阶段 Ray 是主要 async cotrain 路线，
   no-Ray 可同步。
3. Gate 7.3 的 classifier real-data `cls/f1 >= 0.6` 与 Gate 8 的 GPU 6/7 full run
   尚未运行，因此不能声明 pipeline-valid 或 performance-improving。
