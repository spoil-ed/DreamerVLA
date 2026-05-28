# 多进程 Collector + Batched Encoder/Learner 方案

## 背景

当前在线 PPO 的瓶颈主要不是 PPO 本身，而是真实环境采样阶段：

- `third_party/LIBERO/robosuite env.step()` 是串行模拟器调用。
- VLA encoder 每个真实 env step 都要把两路图像、state、task description 编成 `35*1024` hidden。
- 已经修成真实执行 5-step chunk 后，actor forward 从每 step 一次降到每 5 step 一次，但 collect fps 仍大约 `1.0-1.2 env-step/s`。
- 实验版“一个进程里串行多个 env + batched encoder”能跑，但没有明显提速，因为多个 env 仍然在同一个 Python 进程里串行 `env.step()`。

所以真正可能提速的方向是：让多个 collector 进程并行跑真实 env，再把 obs 汇总到一个 encoder/learner 进程做 batched VLA encoder 和策略推理。

## 目标

实现一个不覆盖当前训练脚本的实验版 pipeline：

1. 每张 GPU 一个 learner/encoder 进程。
2. 每张 GPU 对应多个 CPU collector 进程。
3. Collector 只负责真实环境交互、episode/reset、发送 obs、接收 action chunk。
4. Learner 批量收集多个 collector 的 obs，一次过 VLA encoder，批量更新 WM latent，批量出 policy action chunk。
5. Collector 连续执行收到的 5-step chunk，并把每个真实 step 的 transition 发回 learner。
6. Learner 写 replay，按原 WMPO/PPO 逻辑训练 WM、policy、classifier。

## 进程拓扑

推荐第一版拓扑：

```text
torchrun rank 0 on GPU 4
  learner_rank0
    collector_0_0  -> task 0,4,8...
    collector_0_1  -> task 1,5,9...
    collector_0_2  -> optional

torchrun rank 1 on GPU 5
  learner_rank1
    collector_1_0  -> task 2,6...
    collector_1_1  -> task 3,7...
    collector_1_2  -> optional
```

DDP 仍然只包 learner 上的 policy / WM / classifier。Collector 不持有模型，也不碰 CUDA。

## Collector 职责

每个 collector 是独立 Python process，持有一个 `DreamerVLAOnlineTrainEnv`。

它维护：

- `env`
- 当前 `obs`
- 当前 episode buffer
- task cycling
- pending action chunk
- episode return/len
- chunk id/index

它不做：

- VLA encoder
- world model encode/observe
- policy forward
- PPO/WM/classifier update

Collector 的主循环：

```text
reset env
send EncodeRequest(obs, collector_id, episode_state)
wait ActionChunkResponse(actions[5])
for k in 0..4:
    env.step(actions[k])
    send Transition(obs_before_step, action[k], wm_action, reward, done, info)
    if done:
        send EpisodeEnd(episode metadata)
        reset env
        break
    else:
        send EncodeRequest(next_obs)
        wait next ActionChunkResponse only after current chunk consumed
```

注意：collector 发送给 learner 的 obs 必须包含：

- `vla_record` 或构造它所需的 frame history/state/task_description
- `image` 或 replay 所需 image 字段
- `task_id`
- `is_first/is_last/is_terminal`

第一版可以直接 pickle 当前 obs dict，先不做极限优化。

## Learner 职责

Learner 是当前在线训练脚本的核心迁移对象。

它维护：

- encoder
- world_model
- policy/ref_policy
- classifier
- replay
- 每个 collector 对应的 latent
- 每个 collector 对应的 previous wm action
- 每个 collector 的 pending metadata

Learner 的采样循环：

```text
while env_step < total_env_steps:
    gather pending EncodeRequest from collectors
    batch obs list
    obs_embeddings = batched VLA encoder(obs list)

    for each collector request:
        update/initialize its WM latent:
            if is_first: encode_latent(obs_embedding)
            else: observe_next(latent, obs_embedding, prev_wm_action)

    batch actor_input from all ready collector latents
    action_chunks = policy.sample(batch_actor_inputs, return_chunk=True)

    send each collector its own action chunk

    drain Transition messages:
        append per-step records to per-collector episode buffer
        update prev_wm_action for collector
        if EpisodeEnd:
            replay.add_episode(...)

    run readiness checks and WM/PPO/classifier updates as before
```

关键点：真实 env 仍然按每个 step 记录 transition；只是 encoder 和 policy 在 learner 端批处理。

## 通信协议

第一版用 `torch.multiprocessing` 或 Python `multiprocessing` 的 queue 即可。

建议消息类型：

```python
EncodeRequest:
    collector_id: int
    obs: dict
    is_first: bool
    prev_wm_action: np.ndarray | None

ActionChunkResponse:
    collector_id: int
    chunk_id: int
    actions: np.ndarray  # [K, 7], policy normalized scale

TransitionMsg:
    collector_id: int
    obs: dict
    policy_action: np.ndarray
    wm_action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    chunk_id: int
    chunk_index: int
    chunk_len: int

EpisodeEndMsg:
    collector_id: int
    task_id: int
    success: bool
    episode_len: int
    episode_return: float
```

为了避免 queue 堵塞，第一版可以用：

- collector -> learner: one shared `request_queue`
- learner -> collector: 每个 collector 一个 `response_queue`

## Replay 设计

Replay 仍在 learner 进程里，保持现在 `OnlineReplay` 的语义：

- 成功轨迹全 episode 可采样。
- 失败轨迹只用 prefix 策略。
- task-balanced replay 保留。
- episode metadata 继续记录：
  - collector_id
  - rank
  - task_id
  - episode_id
  - collection_index
  - task_episode_index
  - chunk index counts
  - success/failure

不建议把 replay 放 collector 里，因为 PPO/WM/classifier update 都在 learner 里，跨进程采 replay 会复杂很多。

## DDP 同步

每个 rank 一个 learner。每个 learner 本地挂多个 collector。

DDP 逻辑保持：

- learner rank 内部先收集本地 replay。
- readiness 继续用 `get_replay_task_stats_global` 聚合各 rank replay stats。
- PPO/WM/classifier update 时，所有 learner rank 同步执行同数量 update。

Collector 进程不参与 DDP barrier。

## 速度预期

这个方案才可能真正提速，因为：

- 多个 env.step 在多个 CPU process 并行。
- 多个 obs 合成一个 batch 过 VLA encoder。
- 多个 latent 合成一个 batch 过 policy。

初始建议：

- 每 rank 2 collectors：保守，先看稳定性。
- 每 rank 4 collectors：如果 CPU 和内存撑得住，再尝试。

预期：

- 2 collectors/rank：可能 `1.5x-2x`。
- 4 collectors/rank：可能 `2x-3x`，但 reset/env CPU 竞争会更明显。

如果 batch encoder 本身不是瓶颈，而 robosuite/reset 是主瓶颈，收益会低一些。

## 风险和约束

1. Queue 死锁
   - Collector 等 action，learner 等 obs，很容易因为异常没广播 stop signal 卡住。
   - 需要统一 `StopMsg` 和异常传播。

2. Episode/replay 顺序
   - 多 collector 并行后 episode 到达顺序不再等于 task 顺序。
   - replay 必须依赖 metadata，不依赖自然顺序。

3. GPU batch 尺寸不稳定
   - 不同 collector 会在 reset、done、env.step 上耗时不同。
   - learner 可以用短 timeout 聚合请求，例如等到 `min_batch=2` 或 `timeout=20ms` 就 encode。

4. Memory
   - 多 collector 会多持有 raw obs/image/frame history。
   - 第一版先用 pickle obs，后续再优化共享内存或只传必要字段。

5. Task coverage
   - collector 初始化 task offset 必须错开，否则多个 collector 采同一批 task。
   - 推荐 `task_id = task_ids[(rank * collectors_per_rank + collector_id) % len(task_ids)]`，后续按 global collector stride 轮转。

## 实现阶段

### Phase 1: Collect-only smoke

新增实验文件，不动当前正式脚本：

- `scripts/training/train_online_pi0_action_hidden_dreamervla_multiproc.py`
- `scripts/smoke/run_multiproc_collector_smoke_g45.sh`

只验证：

- 多 collector process 能 reset/step。
- learner 能 batched encode。
- action chunk 能返回 collector。
- episode 能进入 replay。
- no PPO/no WM update。

### Phase 2: WM refresh smoke

打开 WM update，但不 PPO：

- replay readiness 正常。
- WM loss 可计算。
- DDP rank update 数一致。
- classifier 不更新。

### Phase 3: Full PPO smoke

打开：

- WM refresh
- PPO outcome update
- classifier online update
- detailed logging

小 budget：

- 2 GPU
- 2 collectors/rank
- 4 tasks
- 200-500 env steps
- 20-50 PPO updates

### Phase 4: All-task training

确认无死锁和 replay metadata 正确后，再切到 10 tasks：

- 2 collectors/rank 起步。
- 如果稳定，再试 4 collectors/rank。

## 日志要求

必须记录：

- `collector_id`
- `rank`
- `task_id`
- episode success/failure
- episode length
- chunk index counts
- request queue lag
- encoder batch size
- encoder batch latency
- env step latency per collector
- reset latency
- learner update latency
- replay task stats
- global replay task stats

这样才能区分到底是：

- env 慢
- reset 慢
- encoder 慢
- queue 等待慢
- PPO/WM update 慢

## 推荐结论

不要把当前“串行多 env”版本作为正式提速方案。它主要证明了 multi-env metadata/replay/chunk 逻辑可行。

真正值得实现的是本方案：collector 多进程并行 env，learner 单点 batched encoder/policy/PPO。第一版实现 collect-only smoke 就足够验证核心收益。如果 collect fps 不能明显超过当前 `1.0-1.2 env-step/s`，说明瓶颈主要是 robosuite/reset 或 CPU 资源，不应该继续复杂化 PPO 训练端。
