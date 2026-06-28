# DreamerVLA Online Cotrain Pipeline Explained

本文档解释 DreamerVLA 当前 **cold-start collection -> offline warmup -> online cotrain**
训练链路，重点覆盖 OpenVLA-OFT action-hidden / LUMOS 路线。目标是把真实代码中的
`step`、`episode`、`rollout`、`replay`、world model warmup、classifier warmup、
imagined PPO 更新、Ray overlap、成功率统计和关键指标一次讲清楚。

本文档不是论文方法概述，而是工程语义手册。每个阶段都对应当前代码路径和配置入口。
如果目的是逐项审查 pipeline 是否正确，请参考
[bracket resolution checklist](../reports/audits/online_cotrain_bracket_resolution_checklist.md)。

## 0. 总览

完整 release pipeline 由一个 launcher 编排：

```text
scripts/e2e_coldstart_warmup_cotrain_ray.sh
  -> dreamervla.launchers.coldstart_warmup_cotrain
      PHASE 1: cold-start rollout collection
      PHASE 2a: offline WM + classifier warmup
      PHASE 2b: consolidate warmup ckpts for Ray async
      PHASE 2c: Ray online cotrain
```

核心思想：

```text
真实 LIBERO episode
  -> 存成 reward HDF5 + hidden sidecar HDF5
  -> seed 到 OnlineReplay
  -> world model 学真实 hidden dynamics
  -> classifier 学真实 success / failure 判别
  -> online rollout 继续把真实 episode 放进 replay
  -> learner 从 replay 采样
      -> 更新 world model
      -> 更新 classifier
      -> 在 world model 里想象多条轨迹
      -> classifier 给想象轨迹打成功分
      -> GRPO/PPO 更新 actor
```

必须区分两条成功率：

| 名称 | 来源 | 何时更新 | 含义 |
| --- | --- | --- | --- |
| `rollout/success_rate` | 真实 LIBERO env 的 `info["success"]` | 真实 episode done 时 | 环境真实成功率 |
| `LUMOS/success_rate` | classifier 对 WM imagined latent video 的 `predict_success` | PPO learner update 时 | 想象轨迹被 classifier 判定成功的比例 |

二者不能互相替代。`rollout/success_rate` 为 0 说明真实环境 episode 没成功，或者还没有
episode 结束；`LUMOS/success_rate` 为 0 说明 classifier 在想象轨迹里没有判出成功。

## 1. 代码和配置入口

| 责任 | 文件 / 配置 |
| --- | --- |
| 顶层编排 | `dreamervla/launchers/coldstart_warmup_cotrain.py` |
| launcher 参数 | `configs/scripts/coldstart_warmup_cotrain.yaml` |
| offline warmup runner | `dreamervla/runners/online_cotrain_pipeline_runner.py` |
| no-Ray online runner | `dreamervla/runners/online_cotrain_runner.py` |
| Ray online runner | `dreamervla/runners/online_cotrain_ray_runner.py` |
| Ray replay actor | `dreamervla/workers/replay/replay_worker.py` |
| Ray env actor | `dreamervla/workers/env/env_worker.py` |
| Ray OFT rollout inference actor | `dreamervla/workers/inference/rollout_inference_worker.py` |
| Ray learner actor | `dreamervla/workers/actor/learner_worker.py` |
| replay buffer | `dreamervla/runners/online_replay.py` |
| offline seed | `dreamervla/runners/offline_seed.py` |
| WM update step | `dreamervla/algorithms/dreamervla.py::world_model_pretrain_step` |
| classifier update step | `dreamervla/runners/online_dreamervla.py::online_classifier_update_step` |
| LUMOS / PPO update | `dreamervla/algorithms/ppo/outcome.py::dino_lumos_step` |
| current OFT chunk WM | `dreamervla/models/world_model/dino_wm_chunk.py` |
| success classifier | `dreamervla/models/reward/latent_success_classifier.py` |
| pipeline experiment | `configs/experiment/online_cotrain_pipeline_oft_action_hidden.yaml` |
| pipeline config body | `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` |
| Ray async experiment | `configs/experiment/online_cotrain_ray_oft_action_hidden.yaml` |
| Ray async config body | `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml` |

## 2. 核心术语

### Env step

一个真实环境 transition：

```text
obs_t + action_t -> obs_{t+1}, reward_t, done_t, info_t
```

在 Ray runner 中，`rollout.steps` 现在表示真实 env transition 数量，不是 inference batch 数量。

### Inference batch

一次 batched policy / OFT rollout bundle 前向，可以同时给多个 env worker 产出动作。
如果 `env.num_workers=4`，一个 inference batch 可能产出 4 个 env action，但真实
`rollout/steps` 增加的是 env step 数量。

### Action chunk

VLA actor 一次决策输出 `K` 个连续 env actions：

```text
action_chunk: [K, action_dim]
```

当前 OFT action-hidden 路线中：

```text
K = task.legacy_action_hidden.chunk_size
```

常见配置里 OFT `K=8`。在真实 env rollout 中，inference worker 会把 action chunk 放入
per-env queue，然后每个 env step 弹出一个 action 执行。在 LUMOS imagination 中，一个
actor decision 也是一个 chunk，world model 一次 `predict_next_chunk` 前进 `K` 个 env steps。

### Episode

从 `env.reset()` 到 `terminated or truncated` 的完整真实轨迹。只有 episode 结束时，
真实成功率才有新的样本。

### Rollout

在本文中有两个语境：

1. 真实 rollout：LIBERO env 中执行 policy / OFT bundle 得到真实 episode。
2. 想象 rollout：从 replay latent start 出发，在 world model 内部滚动，不接触真实 env。

### Replay window

learner 从 `OnlineReplay` 中采样的固定长度窗口：

```text
[obs_embedding, actions, rewards, dones, is_first, is_terminal, is_last]
```

`sequence_length` 控制窗口长度。WM warmup、classifier warmup 和 PPO imagination 都从
这个 replay buffer 取数据。

### Learner update

一次 `LearnerWorker.update("cotrain", 1)` 默认依次执行：

```text
wm update -> classifier update -> rl/PPO update
```

它不是 env step，也不是 episode。Ray overlap 模式下，learner update 可以和新的 env rollout
并行发生。

### PPO epoch

`algorithm.ppo_update_epochs` 是对同一批 imagined rollout 做几轮 PPO re-evaluation。
它不是 dataset epoch，也不是环境 episode。

### GRPO group

同一个 replay start 会重复想象 `algorithm.ppo_rollouts_per_start` 条 rollout。它们形成一组：

```text
group_size = ppo_rollouts_per_start
```

组内 return 有方差时，GRPO advantage 才有非零学习信号。`group_size=1` 时组内方差恒为 0，
如果启用 zero-variance filter，PPO 信号会被过滤掉。

## 3. 数据表示

### `obs_embedding`

`obs_embedding` 是当前 world model 直接使用的 compact hidden 表征。对 OFT action-hidden
路线，它来自 VLA/OFT action-query hidden 的 sidecar，通常已经 flatten 成一个 frame-level
hidden：

```text
obs_embedding_t: [wm_obs_dim]
```

world model warmup、classifier warmup、LUMOS imagination 都主要围绕这个表示工作。

### `actor_hidden_states`

`actor_hidden_states` 是更完整的 actor token hidden 序列。它比 `obs_embedding` 信息更细，
但不是当前 OFT chunk WM 的核心输入。只有 world model 带 `sequence_decoder` 且 batch 中
真的包含 `actor_hidden_states` 时，`full_hidden_*` 指标才有意义。

### `actions` 和 `wm_action`

真实 env 执行动作和 WM 条件动作必须处在同一 scale 语义下。当前 OFT Ray rollout path：

```text
OFT rollout bundle -> process_action -> env.step(action)
EnvWorker transition -> wm_action = action
Replay sample -> actions/current_actions
WM chunk_loss / LUMOS imagination condition on wm_action
```

配置里 `algorithm.rssm_action_scale: env` 表示 actor action 传给 world model 前按 env scale
处理。若 env 期待 normalized action 但 WM 存 raw action，或者反过来，就会出现双重缩放或
尺度错配，真实成功率可能直接归零。

### Success fields

离线 seed 阶段会从 HDF5 中读取：

- `sparse_rewards`
- `dones`
- `demo.attrs["episode_success"]`

并转换成 replay transition 的：

- `reward`
- `done`
- `is_last`
- `is_terminal`
- `success`

`OnlineReplay` 判定 episode success 时会看：

```text
step["success"] or is_terminal > 0.5 or reward > 0
```

真实 Ray rollout 中，episode 结束时的统计来自 env 返回的 `info["success"]`。

## 4. Stage 1: Cold-Start Collection

入口：

```text
dreamervla.launchers.coldstart_warmup_cotrain
  PHASE 1/2 START: cold-start collection
```

常用配置：

```text
mode=ray
task=goal
profile=multi_gpu
skip_collect=false
```

collection 的产物在统一 collected-rollouts 目录：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/
  reward/*.hdf5
  hidden/*.hdf5
  collection_manifest.json
  resolved_config.yaml
```

Ray collection / Ray OFT online rollout 的 inference 逻辑是：

```text
EnvWorker.current_obs()
  -> RolloutInferenceWorker.forward_batch(obs_batch, env_ids)
      -> OFTRolloutBundle.predict_batch(<prepared_observations>)
      -> action_chunk + flat_hidden
      -> per-env action queue
      -> process_action(action)
      -> obs_embedding = flat_hidden
  -> EnvWorker.step(action, obs_embedding)
      -> env.step(action)
      -> transition includes obs_embedding
      -> if done: replay.add_episode(episode)
```

collection 阶段会把真实 episode 同步写成 reward HDF5 和 hidden HDF5。warmup 后续不重新
跑 VLA encoder，而是直接读取这些 sidecar。

### Collection success

collection 里的成功是 LIBERO env 的真实成功，不是 classifier 预测。episode 是否成功会写入
HDF5 attrs 或 sparse reward。这个真实标签是 classifier warmup 的监督来源。

## 5. Stage 2: Offline Seed Into Replay

入口：

```text
OnlineCotrainPipelineRunner.run()
  -> seed_replay_from_offline(<reward_dir>, <hidden_dir>)
```

`seed_replay_from_offline` 会将 reward shard 和 hidden sidecar 按同名 shard/demo 配对：

```text
reward/ray_shard_000.hdf5::data/demo_*
hidden/ray_shard_000.hdf5::data/demo_*/obs_embedding
```

每个 demo 会变成一条 replay episode。每个 transition 大致包含：

```text
{
  "image": agentview_rgb[t],
  "obs_embedding": hidden[t],
  "reward": sparse_reward[t] or success fixup,
  "done": dones[t],
  "is_last": dones[t],
  "is_terminal": step_success,
  "wm_action": actions[t],
  "task_id": task_id,
  "success": step_success,
}
```

### Full warmup vs capped warmup

`offline_warmup.max_episodes_per_task` 控制最多 seed 多少条 episode：

| 设置 | 含义 |
| --- | --- |
| `null` | 不按 task 截断，能读到多少 episode 就加入多少 |
| `10` | 每个 task 最多加入 10 条 episode |

但这还不够。`OnlineReplay` 还有容量限制：

```text
online_rollout.buffer_size
online_rollout.replay_capacity_mode
```

当前 release / multi-GPU 配置使用：

```text
online_rollout.replay_capacity_mode: total_sharded
online_rollout.buffer_size: 160000
```

`total_sharded` 下，每个 task 的容量大约是：

```text
buffer_size / number_of_task_ids
```

因此“full warmup”实际需要两个条件同时满足：

1. `offline_warmup.max_episodes_per_task: null`
2. `online_rollout.buffer_size` 足够容纳所有 seed transitions

否则日志可能显示读到了 500 episodes，但 replay 由于容量驱逐只保留很少 transitions。

## 6. Stage 3: Offline World Model Warmup

入口：

```text
OnlineCotrainPipelineRunner._offline_warmup_wm
  -> replay.sample(batch_size)
  -> _build_wm_pretrain_batch(<replay_batch>)
  -> world_model_pretrain_step(<wm_batch>)
```

`world_model_pretrain_step` 做的事：

```text
batch -> move tensors to device
world_model.train()
losses = world_model(batch)
loss = losses["_loss"] or losses["loss"]
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(world_model.parameters())
optimizer.step()
return scalar metrics
```

当前 OFT action-hidden route 使用：

```text
ChunkAwareDinoWMWorldModel.chunk_loss
```

它的训练目标是 chunk dynamics：

```text
history = obs_tokens[:, :H]
chunk_actions = actions[:, H-1 : H-1+K]
hidden_target = obs_tokens[:, H : H+K]

hidden_pred = predict_next_chunk(history, chunk_actions)["hidden_seq"]
loss = hidden reconstruction loss
```

其中：

- `H = world_model.num_hist`
- `K = world_model.chunk_size`
- `hidden_target` 是未来 `K` 个 env step 的 `obs_embedding`
- `hidden_pred` 是 WM 根据历史 hidden 和 action chunk 预测出的未来 hidden

如果启用 `chunk_rollout_chunks > 1`，WM 还会做 closed-loop anti-drift loss：

```text
第 1 个 chunk 用真实 history
第 2..N 个 chunk 用上一个 chunk 的预测 hidden 继续滚动
actions 仍然来自真实 demo
```

这能约束 WM 在多 chunk 想象时不要快速漂移。

### WM hidden metrics

| 指标 | 目标 | 形式 | 用途 |
| --- | --- | --- | --- |
| `hidden_rec_loss` / `hidden_mse` | compact `obs_embedding` | MSE | 判断 WM latent 是否能数值恢复当前 pipeline 实际使用的 hidden |
| `hidden_cosine_loss` | compact `obs_embedding` | `1 - cosine` | 判断语义方向是否对齐，较少受范数影响 |
| `full_hidden_rec_loss` | `actor_hidden_states` 完整 token hidden 序列 | masked MSE | 只对启用 sequence decoder 且 batch 有完整 actor hidden 的 WM 有意义 |
| `full_hidden_cosine_loss` | `actor_hidden_states` 完整 token hidden 序列 | masked cosine | 同上，检查完整 token 序列方向对齐 |

当前 OFT chunk WM (`dino_wm_chunk.py`) 主要产出：

```text
hidden_mse
hidden_cosine_loss
rollout_loss
rollout_mse
rollout_cosine_loss
reward_loss
success_return_loss
```

它不产出 `full_hidden_*`。因此当前训练不要用 `full_hidden_*` 判断 WM 是否学到东西。

### WM warmup checkpoint

warmup 结束后保存：

```text
${training.out_dir}/ckpt/wm_warmup.ckpt
${training.out_dir}/ckpt/wm_warmup_hf/   # if checkpoint_format includes hf
```

## 7. Stage 4: Offline Classifier Warmup

入口：

```text
OnlineCotrainPipelineRunner._offline_warmup_classifier
  -> online_classifier_update_step(<classifier_batch>)
```

更新逻辑：

```text
replay.sample_classifier_windows(<batch_size>, <window>, <chunk_size>)
  -> windows: [B, window, latent_dim]
  -> labels: 0/1
classifier(windows)
cross_entropy(logits, labels)
backward + grad clip + optimizer.step
```

### Classifier labels

`OnlineReplay.sample_classifier_windows` 会从完整 episode 中采样窗口。

对于成功 episode：

- 终点窗口，也就是接近 `finish_step` 的窗口，可以标为正样本。
- 更早的窗口按 `early_neg_stride` 采样，标为负样本。

对于失败 episode：

- 可采样窗口标为负样本。

这使 classifier 学的是：

```text
一段 latent window 是否已经包含足够证据表明任务成功
```

而不是简单记住 episode id。

### Chunk granularity

当前 classifier 配置：

```text
classifier.granularity: chunk
classifier.chunk_size: K
classifier.chunk_pool: last
classifier.window: 8
```

含义：

1. 原始 env-step hidden 会按每 `K` 个 step 聚合成一个 chunk hidden。
2. `chunk_pool=last` 表示取每个 chunk 的最后一个 hidden。
3. classifier 的一个 window 包含 `classifier.window` 个 chunk。

因此 classifier 的 native time unit 是 chunk，不是 env step。LUMOS imagination 会在边界处
把 classifier 返回的 `finish_step` 从 chunk index 映射回 env-step index：

```text
finish_env_step = finish_chunk * K + (K - 1)   # chunk_pool=last
```

### Classifier metrics

| 指标 | 含义 |
| --- | --- |
| `cls/loss` | cross entropy |
| `cls/acc` | batch accuracy |
| `cls/f1` | 正类 F1，最重要的非退化指标 |
| `pos_frac` | batch 中正样本比例 |
| `prob_mean` | 正类概率均值 |
| `grad_norm` | classifier 梯度范数 |

warmup 阶段日志中会看到：

```text
[pipeline][cls-warmup] step=<i>/<steps> loss=<loss> acc=<acc> f1=<f1> pos=<pos_frac>
```

如果 `cls/f1` 长期接近 0，classifier 没有给 PPO 提供可靠成功判别。

### `predict_success`

PPO imagination 使用：

```text
LatentSuccessClassifier.predict_success(latent_video, threshold, stride, min_steps)
```

它会滑窗扫描 imagined latent video，返回：

| 返回值 | 含义 |
| --- | --- |
| `complete` | 是否有窗口概率超过 threshold |
| `finish_step` | 第一个超过 threshold 的窗口终点 |
| `score` | 全部扫描窗口中的最大 `p(success)` |
| `score_step` | 最大 score 出现的位置 |

当前 LUMOS reward model 配置为：

```text
algorithm.lumos.reward_model: probability_outcome
```

也就是说即使 threshold 下全是失败，`score` 的连续概率也仍能提供一些排序信号；但如果同一
GRPO group 内 `score` 也几乎没有方差，advantage 仍然接近 0。

### Classifier warmup checkpoint

warmup 结束后保存：

```text
${training.out_dir}/ckpt/classifier_warmup.ckpt
${training.out_dir}/ckpt/classifier_warmup_hf/   # if checkpoint_format includes hf
```

## 8. Stage 5: Warmup To Online Handoff

如果 `cotrain_engine=sync`，launcher 直接运行：

```text
experiment=online_cotrain_pipeline_oft_action_hidden
```

这个 runner 在同一个进程中完成 warmup 后进入 no-Ray online cotrain loop。

如果 `cotrain_engine=async`，launcher 分三步：

```text
PHASE 2a/3: offline warmup (sync, writes warmup ckpts)
PHASE 2b/3: consolidate warmup ckpts -> ray async init
PHASE 2c/3: async online cotrain (ray overlap)
```

consolidate 会把：

```text
wm_warmup.ckpt
classifier_warmup.ckpt
```

合成 Ray runner 可读取的：

```text
ray_async_init.ckpt
{
  "state_dicts": {
    "world_model": "<world_model_state_dict>",
    "classifier": "<classifier_state_dict>"
  }
}
```

然后 Ray async online runner 用这个 checkpoint 初始化 learner。

## 9. Stage 6: Ray Online Cotrain Loop

Ray async online 入口：

```text
experiment=online_cotrain_ray_oft_action_hidden
  -> OnlineCotrainRayRunner
```

它会创建以下 Ray actors：

| Actor | 作用 |
| --- | --- |
| `ReplayWorker` | 包装 `OnlineReplay`，保存完整 episode，提供 sample |
| `EnvWorker` | 持有 LIBERO env，执行真实 `env.step` |
| `RolloutInferenceWorker` 或 `InferenceWorker` | 根据 obs 产出 action 和 `obs_embedding` |
| `LearnerWorker` | 从 replay 采样，执行 WM/classifier/RL 更新 |
| weight store / syncer | learner policy 权重同步给 inference worker |

### Sync loop

同步 loop 的顺序是：

```text
current_obs
-> infer.forward_batch
-> env.step for each env
-> if replay ready: learner.update
-> sync policy weights
```

它容易理解，但 rollout 和 learner 串行，资源利用率低。

### Overlap loop

async overlap loop 用 Ray `ObjectRef` 管理三个队列：

```text
ready_obs
pending_infers
pending_steps
pending_learn
```

逻辑是：

```text
1. 有 ready_obs 且还有目标 env steps 未完成 -> launch infer
2. infer 完成 -> launch env.step
3. env.step 完成 -> env_steps += 1
4. 如果 done -> episode 加入 replay，记录真实 success
5. replay ready 且无 pending learner -> launch learner.update("cotrain", 1)
6. learner 完成 -> log metrics，push policy weights
7. inference worker 支持的话 pull policy weights
8. 重复直到 rollout.steps 个真实 env step 完成
```

关键点：

- 每个 env worker 内部仍然严格串行，不会同一个 env 并发 step。
- 多个 env worker 可以并行。
- learner update 可以和正在进行的 inference / env step 重叠。
- episode 只有 done 时才进入 replay。
- 真实成功率只有 done 时才更新。

### 当前 Ray OFT async 的重要实现事实

当前 OFT Ray async rollout 使用：

```text
RolloutInferenceWorker
  -> OFTRolloutBundle
```

这个 worker 的 `pull_weights(<store>, <key>, <version>)` 是 no-op。原因是当前 OFT online rollout 仍由固定 OFT
base policy / rollout bundle 驱动；learner 里的 trainable actor 在 world model imagination
中更新。

因此在当前 **Ray OFT async** 路线中：

```text
PPO actor update 会改变 imagined actor
但不会立即改变真实 env rollout policy
```

这意味着：

- `rl/ppo_step_applied` 和 `rl/policy_grad_norm` 可以证明 PPO 在学。
- `LUMOS/score_mean`、`LUMOS/score_std` 可以证明想象打分在变化。
- 但 `rollout/success_rate` 不一定随 actor update 立刻提升，因为真实 rollout 仍是 OFT base policy。

若目标是“真实 env rollout 由更新后的 actor 驱动”，需要使用支持 policy weight pull 的
`InferenceWorker` 路线，或继续把 OFT Ray rollout worker 改成 actor-driven。no-Ray
`OnlineCotrainRunner` 的 actor-driven rollout 语义需要和 Ray OFT async 分开看。

## 10. Stage 7: Learner Cotrain Update

Ray learner 入口：

```text
LearnerWorker.update("cotrain", 1)
```

`phase="cotrain"` 会展开为：

```text
wm -> classifier -> rl
```

### 10.1 WM update

```text
LearnerWorker._dreamervla_wm_update_once
  -> replay.sample(batch_size)
  -> world_model_pretrain_step
  -> returns {"wm/loss": <loss>}
```

目前 Ray learner 只透传 `wm/loss`。`world_model_pretrain_step` 内部能返回更多 WM
diagnostics，但 Ray learner 没有全部 log 出来。如果要在 TensorBoard 里看到
`hidden_rec_loss`、`hidden_cosine_loss`，需要扩展 `_dreamervla_wm_update_once` 的返回值。

### 10.2 Classifier update

```text
LearnerWorker._dreamervla_classifier_update_once
  -> online_classifier_update_step
  -> returns cls/loss, cls/acc, cls/f1
```

classifier update 会更新 learner 内部记录：

```text
_cotrain_classifier_updates
_cotrain_last_classifier_f1
```

如果启用 `actor_signal_gate`，RL update 可以等 classifier 达到指定 F1 或更新次数后再开始。
当前默认 gate 是关闭的，因为 LUMOS/PPO 内部已经会根据 group variance 和 mask 决定是否
真正 `optimizer.step()`。

### 10.3 RL / PPO update

```text
LearnerWorker._dreamervla_rl_update_once
  -> replay.sample(batch_size)
  -> dino_lumos_step(<policy>, <world_model>, <classifier>, <replay_batch>)
```

传给 PPO 的 replay batch 包含：

```text
obs_embedding
actions
rewards
dones
is_first
is_terminal
is_last
```

PPO 不是在真实 env 里再跑一遍，而是在 world model 里想象。

## 11. LUMOS / PPO 的完整过程

入口：

```text
dreamervla.algorithms.ppo.outcome.dino_lumos_step
```

关键配置：

```text
algorithm.lumos.chunk_size = K
algorithm.lumos.episode_max_steps = T_max
algorithm.imag_last
algorithm.ppo_rollouts_per_start
algorithm.ppo_update_epochs
algorithm.lumos.update_micro_batch_starts
algorithm.lumos.filter_zero_variance_groups
algorithm.lumos.reward_model
```

### 11.1 从 replay window 选 start states

先让 world model observe replay sequence：

```text
latent_seq = world_model({"mode": "observe_sequence", **obs})
```

然后从这个 sequence 中选 `imag_last` 个 start points。当前逻辑不是简单取最后几个相邻
frame，而是在有效历史范围内 strided 选择，使 start phase 更分散。

每个 start 重复 `ppo_rollouts_per_start` 次：

```text
n_starts = batch_size * imag_last
B_eff = n_starts * ppo_rollouts_per_start
```

这些重复 rollout 形成 GRPO groups。每组来自同一个 start state。

### 11.2 想象 full episode

令：

```text
num_chunks = T_max // K
```

对每个 chunk：

```text
actor_feat = world_model.actor_input(current_latent)
action_chunk, old_log_prob = policy.sample(actor_feat)
wm_action_chunk = _actor_action_for_world_model(action_chunk, algorithm_cfg)
next_seq = world_model.predict_next_chunk(current_latent, wm_action_chunk)
video_latents.append(next_seq.hidden_seq or pooled chunk hidden)
current_latent = next_seq.last_latent
```

这一步在 `no_grad` 中进行。它生成的是固定的 rollout 数据，不让梯度穿过 world model
dynamics。PPO 的梯度后面通过 policy re-evaluate 固定 action 来算。

为了节省显存：

- `update_micro_batch_starts` 按 start group 切片。
- `imagine_micro_batch` 限制 world model forward 的 rollout batch。
- `eval_micro_batch` 限制 classifier scan 的 batch。
- actor features 会临时放到 CPU，PPO re-eval 时再搬回 GPU。

### 11.3 Classifier 给想象轨迹打分

对想象出来的 `video_latents`：

```text
classifier.predict_success(video_latents, threshold, stride=1, min_steps=<min_steps>)
```

得到：

```text
complete
finish_step
score
score_step
```

如果 classifier 是 chunk-granular，`finish_step` 会从 chunk unit 映射回 env-step unit。

### 11.4 Reward model 生成 return

当前配置常用：

```text
algorithm.lumos.reward_model: probability_outcome
```

它用 classifier 的连续 `score` 作为 outcome reward 来源。另一种 `sparse_outcome` 则主要
使用 `complete` 和 `finish_step` 放置 0/1 sparse reward。

最终：

```text
reward_tensor: [B_eff, T_max]
returns = reward_tensor.sum(dim=-1)
```

如果有 reference policy KL：

```text
returns_adjusted = returns - kl_coef * kl_per_rollout
```

### 11.5 GRPO advantage

按 group 计算 advantage：

```text
advantages = group_normalize(returns_adjusted, group_size=ppo_rollouts_per_start)
```

如果一个 group 内所有 rollout return 一样：

```text
std(group_returns) == 0
advantage == 0
```

当 `filter_zero_variance_groups=true` 时，这些 group 会从 PPO mask 中移除。

典型零信号原因：

| 现象 | 原因 |
| --- | --- |
| `ppo_rollouts_per_start=1` | 每组只有一条 rollout，组内方差恒为 0 |
| classifier 所有 score 相同 | return 无方差，advantage 为 0 |
| classifier 全判失败且 reward 是 sparse | returns 全 0 |
| action scale 错误 | imagined / real action 条件不可信，成功率可能归零 |
| PPO mask 全 0 | `ppo_step_applied=0`，actor optimizer 不 step |

### 11.6 PPO re-evaluate and update

PPO 不重新采样轨迹，而是对刚才固定的 imagined trajectory 做 re-evaluate：

```text
new_log_prob = policy.evaluate(actor_feat, fixed_action_chunk)
ratio = exp(new_log_prob - old_log_prob)
ppo_clip = clipped_surrogate(ratio, advantage)
loss = ppo_term - entropy_coef * entropy
```

如果启用 BC anchor：

```text
loss += actor_bc_to_ref_scale * mse(policy_action_chunk, ref_action_chunk)
```

当前实现中：

- PPO signal mask 使用 zero-variance filter 后的 `chunk_mask`。
- BC anchor 使用 finish-only mask，不被 zero-variance filter 移除。
- 若没有 PPO signal 且没有 BC signal，则跳过 `optimizer.step()`。
- DDP 下会 all-reduce step flag，确保所有 rank 一致 step 或一致 skip。

关键返回：

```text
actor_loss
actor_grad_norm
returns_mean
returns_std
score_mean
score_std
ppo_step_applied
LUMOS/group_var_keep_frac
LUMOS/num_mixed_groups
```

判断 PPO 是否真的在学，最小集合是：

```text
rl/ppo_step_applied > 0
rl/policy_grad_norm > 0
rl/returns_std > 0 or LUMOS/score_std > 0
LUMOS/num_mixed_groups > 0
```

## 12. Step、Episode、Update 的排序关系

### 真实 rollout 侧

```text
env step 1
env step 2
env step k
env step N
episode done
  -> replay.add_episode(full_episode)
  -> rollout episode success counter updates
```

真实 episode 没结束前：

- replay 不会收到这条 partial episode。
- `rollout/success_rate` 不会因为这条 partial episode 改变。

### Learner 侧

Ray overlap 中 learner 不是“每个 env step 必然更新一次”，而是：

```text
if replay.ready(min_episodes) and no pending learner:
    launch learner.update("cotrain", 1)
```

`ReplayWorker.ready(min_episodes)` 当前检查 valid full episodes 数量。

因此合理理解是：

```text
完整 episode 进入 replay 后，learner 才有新 episode 可学；
但 learner 可以一边采旧 replay 学，一边 env workers 继续收集新 episode。
```

这就是 RLinf-style overlap：rollout 和 learner 并行，二者通过 replay 和权重同步解耦。

### PPO 侧

一次 PPO update 内部的时间轴：

```text
replay sample window
-> observe latent sequence
-> select imag_last starts
-> repeat each start group_size times
-> imagine full episode in WM
-> classifier score imagined video
-> build returns
-> group-relative advantage
-> PPO epochs over fixed imagined rollout
-> actor optimizer step or skip
```

这里没有真实 env step。

## 13. 当前指标体系

### Rollout metrics

| 指标 | 含义 |
| --- | --- |
| `rollout/steps` | 已完成真实 env transitions |
| `rollout/infer_batches` | inference forward batch 数量 |
| `rollout/episodes` | 已结束真实 episodes |
| `rollout/successes` | 真实成功 episodes 数 |
| `rollout/success_rate` | `successes / episodes`，无 episode 时为 0 |
| `rollout/success_rate_valid` | 是否已有至少 1 个 episode |
| `rollout/current_success_rate` | 最近一个结束 episode 的 0/1 success |
| `rollout/avg_success_rate` | 当前平均真实成功率 |

每次 episode 结束时，console 会输出：

```text
[rollout] episode=<N> success=<0_or_1> avg_success_rate=<mean> window_success_rate=<window_mean>
```

### Learner / train metrics

| 指标 | 含义 |
| --- | --- |
| `train/learner_updates` | learner 完成的 update 次数 |
| `train/ppo_updates` | 兼容旧 dashboard，等于 learner updates |
| `train/rl_loss` | runner 选择的最近 learner loss，可能是 `rl/actor_loss`、`wm/loss` 或 `cls/loss` |

### WM metrics

Ray learner 当前默认只 log：

```text
wm/loss
```

`world_model_pretrain_step` 内部可返回的 WM 诊断包括：

```text
loss
kl_loss
dyn_kl
rep_kl
transition_loss
reward_loss
success_return_loss
success_return_pred_mean
success_return_target_mean
success_return_mse
hidden_rec_loss
hidden_rec_scaled_loss
hidden_cosine_loss
full_hidden_rec_loss
full_hidden_rec_scaled_loss
full_hidden_cosine_loss
hidden_pred_norm
hidden_target_norm
predicted_reward_mean
latent_norm
grad_norm
```

当前 OFT chunk WM 更应关注：

```text
wm/loss
hidden_mse or hidden_rec_loss
hidden_cosine_loss
rollout_loss
reward_loss
success_return_loss
grad_norm
```

但其中很多还没有从 Ray learner 透传到 TensorBoard。

### Classifier metrics

| 指标 | 含义 |
| --- | --- |
| `cls/loss` | classifier cross entropy |
| `cls/acc` | batch accuracy |
| `cls/f1` | 正类 F1 |

warmup console 还会打印 `pos`，即正样本比例。

### RL / PPO metrics

| 指标 | 含义 |
| --- | --- |
| `rl/actor_loss` | PPO actor loss |
| `rl/returns_mean` | LUMOS returns 均值 |
| `rl/returns_std` | LUMOS returns 标准差 |
| `rl/policy_grad_norm` | actor 梯度范数 |
| `rl/ppo_step_applied` | 本次是否执行 actor optimizer step |
| `rl/actor_signal_ready` | actor signal gate 是否允许 RL |
| `rl/skipped_no_signal` | actor signal gate 是否跳过 RL |
| `rl/classifier_f1_gate` | learner 记录的最近 classifier F1 |
| `rl/classifier_updates` | learner 中 classifier 更新次数 |

### LUMOS metrics

| 指标 | 含义 |
| --- | --- |
| `LUMOS/success_rate` | imagined rollout 被 classifier 判成功的比例 |
| `LUMOS/score_mean` | classifier 最大成功概率均值 |
| `LUMOS/score_std` | classifier 最大成功概率标准差 |
| `LUMOS/group_var_keep_frac` | zero-variance filter 后保留的 group 比例 |
| `LUMOS/num_mixed_groups` | 同组内既有成功又有失败的 group 数 |
| `LUMOS/num_all_success_groups` | 全成功 groups |
| `LUMOS/num_all_fail_groups` | 全失败 groups |
| `LUMOS/mean_finish_step` | 成功 imagined rollout 的平均完成 step；无成功时为 -1 |
| `LUMOS/valid_chunk_frac` | PPO loss mask 中有效 chunk 比例 |
| `LUMOS/group_size` | `ppo_rollouts_per_start` |
| `LUMOS/num_chunks` | `episode_max_steps // chunk_size` |

### Time / overlap metrics

| 指标 | 含义 |
| --- | --- |
| `time/rollout_overlap_events` | rollout inference 与其他工作重叠的次数 |
| `time/rollout_strict_overlap_events` | inference launch 时 env step refs 正在 pending 的次数 |
| `time/rollout_infer_ready_batches` | 完成的 inference batches |
| `time/rollout_env_ready_batches` | 完成的 env step batches |
| `time/infer_*_s` | inference worker 内 encode / world_model / policy 耗时 |
| `time/infer_wait_s` | driver 等 inference 的累计时间 |
| `time/env_step_wait_s` | driver 等 env step 的累计时间 |
| `time/learner_wait_s` | driver 等 learner 的累计时间 |
| `time/weight_sync_wait_s` | 权重同步耗时 |
| `time/ray_wait_s` | `ray.wait` 累计等待时间 |

## 14. 最小判断面板

不要一开始看太多指标。先看这几个：

```text
WM warmup:
  wm loss 是否下降

Classifier warmup:
  cls/f1 是否明显高于随机，目标至少 > 0.5，理想 >= 0.6

Ray online真实采集:
  rollout/episodes 是否增长
  rollout/success_rate_valid 是否为 1
  rollout/success_rate 是否不是统计空值

PPO信号:
  LUMOS/score_std > 0
  rl/returns_std > 0
  LUMOS/num_mixed_groups > 0 或 group_var_keep_frac > 0
  rl/ppo_step_applied > 0
  rl/policy_grad_norm > 0
```

如果这些不满足，先不要讨论最终 success rate 是否提升。

## 15. 常见现象和解释

### 15.1 `rollout/success_rate=0`

可能原因：

1. 还没有任何真实 episode done。此时看 `rollout/success_rate_valid`。
2. episode done 了，但全部失败。看 console `[rollout] episode=<N> success=0`。
3. task 本身对 base policy 很难，比如某些 LIBERO task。
4. action scale 错误导致真实执行动作不对。
5. 当前 Ray OFT async 真实 rollout 仍是 fixed OFT base policy，PPO actor 更新不会立刻改变 env rollout。

### 15.2 `LUMOS/success_rate=0`

说明 classifier 对 imagined latent video 没判成功。可能是：

1. classifier 退化，`cls/f1` 低。
2. WM imagination 漂移，classifier 看不到成功模式。
3. reward model 用 sparse outcome，全部 imagined rollout 都低于 threshold。
4. `classifier_min_steps` 或 chunk granularity 映射错误。

### 15.3 `rl/actor_loss=0` 且 `rl/policy_grad_norm=0`

可能是 PPO 没有信号：

1. `ppo_rollouts_per_start=1`。
2. 同组 returns 全一样。
3. zero-variance groups 全被过滤。
4. classifier score 没方差。
5. actor signal gate 禁止了 RL。

看：

```text
rl/ppo_step_applied
rl/returns_std
LUMOS/score_std
LUMOS/group_var_keep_frac
LUMOS/num_mixed_groups
```

### 15.4 WM loss 下降但 PPO 没信号

这说明 world model 的监督预测在学，但 outcome reward 链路还没给 actor 可用 advantage。
下一步看 classifier F1 和 imagined score 方差。

### 15.5 Classifier F1 高但真实 rollout 成功率不动

当前 Ray OFT async 下这不矛盾：PPO actor 在 imagination 里更新，但真实 env rollout 仍由 fixed
OFT rollout bundle 驱动。此时真实成功率主要反映 base policy 和采样任务，而不是 learned actor。

## 16. 合理训练顺序

推荐 release 验证顺序：

```text
1. 确认 collected rollout 数据存在且正负样本都有
2. full offline seed:
     max_episodes_per_task=null
     buffer_size 足够大
3. WM warmup 2000 steps:
     loss 下降，无 NaN/OOM
4. Classifier warmup 2000 steps:
     cls/f1 >= 0.6 更稳
5. Ray online:
     rollout/episodes 增长
     episode success 真实打印
6. Learner cotrain:
     wm/loss 有更新
     cls/f1 有更新
     LUMOS/score_std 或 returns_std 非零
     rl/ppo_step_applied=1
     rl/policy_grad_norm>0
7. 若目标是提高真实 rollout success:
     确认真实 rollout policy 是否真的使用更新后的 actor
```

## 17. Sync 与 Async 的区别

| 路线 | 入口 | 特点 |
| --- | --- | --- |
| sync pipeline | `OnlineCotrainPipelineRunner` | warmup 后在同 runner 内进入 online loop，易理解，overlap 少 |
| async Ray pipeline | `OnlineCotrainRayRunner` | warmup 后合并 ckpt，Ray env/infer/replay/learner 分 actor，rollout 与 learner overlap |

两者共享核心学习语义：

```text
replay -> WM update
replay -> classifier update
replay starts -> WM imagination -> classifier score -> PPO actor update
```

区别主要在系统调度、worker 边界和权重同步。

## 18. AGENTS.md 相关架构约束

这条 pipeline 应保持以下原则：

1. Hydra 是 source of truth。batch size、warmup steps、chunk size、task ids、checkpoint
   paths、runner target 都应来自 config。
2. Runner 负责生命周期：`setup -> execute -> teardown`。
3. Ray 是 optional backend，不是默认拓扑。
4. 模型、dataset、env、worker 应通过 `_target_` / registry / protocol 解耦。
5. 指标应走 runner logger，并使用 `train/`、`rollout/`、`env/`、`time/` 等 namespace。
6. 一个 run 写到一个 root：`${training.out_dir}`。

当前 Ray worker 路线仍有一个结构债：

```text
LearnerWorker / RolloutInferenceWorker 使用 target/_target_ + importlib 的轻量 builder
```

它是 config-driven 的，但还不是标准 `hydra.utils.instantiate(cfg.component)`。如果继续做全量
解耦，应把这些 builder 收敛成统一 `_target_` aware construction helper，并避免 worker 内
对具体 OFT bundle 做 `target.endswith("<oft target>")` 分支。

## 19. 一句话 mental model

当前 DreamerVLA cotrain loop 的核心不是“每个真实 step 立刻 PPO”，而是：

```text
真实 episode 持续进入 replay；
learner 从 replay 中反复训练 WM 和 classifier；
actor 只在 WM 想象世界里通过 classifier outcome score 做 PPO；
真实 rollout success 是 env episode-end 统计；
想象 success/score 是 PPO 的学习信号；
Ray async 让 rollout 和 learner 并行，但不改变这些语义。
```
