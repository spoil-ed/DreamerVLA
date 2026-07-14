# Complete Loop

主线是 OpenVLA-OFT one-trajectory cold start：

```text
collect -> warmup -> online cotrain -> eval
```

## 1. Collect

collection 在真实 LIBERO 中运行当前 VLA/OFT policy，写出 reward shard、hidden sidecar 和
`collection_manifest.json`。

当前入口：

- `experiment=collect_rollouts` -> `RolloutCollectionRunner`（Ray backend）

collection 只负责产生初始真实数据，不负责训练 world model 或 actor。

## 2. Warmup

cotrain 前先用 collected reward + hidden HDF5 shard 训练：

- world model
- classifier/reward model

world model 和 classifier 使用各自独立 runner 训练；cotrain 通过显式 checkpoint
输入加载二者，不在 shell 中复制训练默认参数。

## 3. Online Cotrain

manual Ray 主线维护同一个完整 OpenVLA-OFT policy：

```text
raw image/text/proprio
  -> checkpoint-defined vision backbone + projector
  -> projected visual tokens [token_count, token_dim]
  -> native text/proprio concat + OpenVLA LM/OFT action decoder
  -> action-token distribution -> environment action
```

当前 checkpoint 的 projected visual token 是 `[256,4096]`；维度来自
`task.openvla_oft.hidden_token.*` 和 checkpoint metadata，并非 policy 中写死的
architecture constant。mainline 不在 token 和 action 之间插入随机 Transformer bridge
或 learned action queries。

online cotrain 同时维护三类模型状态：

- Actor/VLA policy：由 ActorGroup 更新。
- Rollout policy replica：由 RolloutGroup 做 no-grad inference，定期从 Actor 同步。
- World model + classifier：从显式 warmup checkpoint 加载并冻结，由 LearnerGroup
  持有状态并在 setup 时同步给 WMEnvWorker。

每个 `failure_imagined_rl` global step 的固定因果顺序是：

1. 在 step 边界 reset 真实 slot、丢弃任何旧 policy 的半条轨迹，再用 step-entry
   `pi_old` 收集恰好 32 条完整真实轨迹并 drain 为 step-local batch。每个
   slot/rollout epoch 只接收第一条终止轨迹，提前成功不会使预算溢出。
2. 把成功/失败轨迹追加进 bounded historical replay，使用 `RealTrajectory.success`
   作为 episode outcome。
3. 从当前与历史 replay 的失败 episode 中重复采样，固定选择 episode 首帧；同一 PPO
   comparison group 的成员共享同一个起点。
4. 使用冻结 WM 做 closed-loop imagined rollout，冻结 classifier 提供 reward。
5. 冻结 encoder，只把 imagined trajectory 送入原生 OpenVLA actor 的 PPO；真实轨迹
   不进入 PPO。
6. 保存完整 VLA、冻结 WM/CLS 状态、actor optimizer、replay 和 classifier threshold，
   再做只读真实评估。

`pi_initial` 只用于监控；每步 trust region 比较 step-entry `pi_old` 与候选 actor
更新，越界时回滚 actor policy 和 optimizer。当前模式不执行 encoder SFT、re-encode、
WM update 或 classifier update。若 replay 中没有失败 episode，该步明确跳过 imagination
和 PPO，不回退到成功轨迹。

Actor PPO 的训练数据来自 imagined rollout 组装出的 trajectory，不应隐式从 replay
或真实 rollout 替代。

## 4. Eval

最终质量以真实 LIBERO eval 为准。默认在 step 0 和之后每 10 个 global step，用
10 个 task、每 task 10 条固定真实轨迹评估完整 checkpoint。该 100-trajectory batch
只用于评估，不写 replay、
不更新参数、也不重新校准 threshold。除真实 success rate 外，同时报告：

- 当前 encoder latent 上 classifier 的 trajectory F1/precision/recall/accuracy、混淆矩阵、
  PR-AUC/ROC-AUC（类别齐全时）；
- WM 从前三帧真实 history 开始、之后完全递归 rollout 的逐 horizon token MSE/cosine；
- 同一个 frozen step-local classifier 在 closed-loop WM latent 上的同组 trajectory 指标。

WMEnv reward 可以提供训练信号，但不能替代真实环境 success rate。

## Data Flow

```text
Real LIBERO collection
  -> reward/hidden shards + manifest
  -> OnlineReplay / warmup datasets
  -> WM + classifier warmup checkpoints
  -> [32 real -> append replay -> failed episode-start anchors]
  -> [frozen WM/classifier imagine -> actor PPO]
  -> complete VLA/WM/classifier checkpoint
  -> fixed read-only real LIBERO + closed-loop diagnostic eval
```

## Main Configs

- Collection experiment：`collect_rollouts`
- Cotrain base experiment：`openvla_onetraj_libero_cotrain`
- Failure-conditioned imagined-RL experiment：`openvla_libero`
- Eval experiment：`eval_cotrain`
- Classifier role：`classifier=dreamer-cls`；具体 model/dataset 由 `task.classifier` 注入
