# Historical Prompt Reference

This file is historical/reference prompt context. It is not current architecture source of truth; current architecture facts live in `spec/README.md`, `00_overview.md` through `05_ray_runtime.md`, and `99_manual_notes.md`.

---

## 给 Codex 的背景提示：Ray 卡死问题与 RLinf 对齐方向

请先理解这次问题的背景，再开始改代码。

之前真正卡住 cotrain 的问题，不是 `WAIT` 日志本身，也不是单纯看到某个日志停在：

```text
RealEnvWorker step 30 start key=0:0
```

然后没有对应的 `step 30 done`。

这些只是表象。表象包括：

* Env/Rollout 之间一直刷 `WAIT`；
* 某个 `rank:slot`，例如 `0:0`，执行到某一步 `start` 后没有 `done`；
* terminal 看起来不再推进，也没有稳定的 global step / 进度条；
* 外层 launcher 最后只看到 `subprocess.CalledProcessError ... returned non-zero exit status 1`。

这里的 `rank:slot` 需要按包含关系理解：

* `rank` 是 WorkerGroup 里的第几个 Worker / Ray actor；
* `slot` 是这个 Worker 内部的第几个并行 env；
* `0:0` 表示 `rank 0` 这个 EnvWorker 里的 `slot 0`。

之前已经确认过一个 Ray 层面的根因：Ray GCS task event 堆积过多，日志中出现过类似：

```text
Max number of tasks event (100000) allowed is reached
```

GCS 是 Ray 的全局控制面存储，负责维护 actor、task、node、object 等控制信息。DreamerVLA 之前把大量 env/rollout 交互暴露成 Ray actor method call / channel put/get，导致 task event 压力过大。GCS 事件堆积到上限后，Ray actor 可能被系统或 Ray runtime 以 `SIGTERM` / `SYSTEM_ERROR` 形式杀掉；driver 收到 actor 死亡后退出，外层才表现为 `CalledProcessError`。

仓库里已经合入过一个缓解补丁：

* `dreamervla/scheduler/worker_group.py`：Worker actor options 里加入 `enable_task_events=False`，减少 Worker actor/task 往 GCS 写事件；
* `dreamervla/scheduler/cluster.py`：默认设置 `RAY_task_events_max_num_task_in_gcs=1000000`，避免残余 task event 太快打满；
* 对应 commit：`108f943 fix: reduce ray task event pressure`。

但是这只是第一层缓解。当前 DreamerVLA 与 RLinf 的关键差异仍然存在：DreamerVLA 的 RealEnv/Rollout 通信是 per-slot Ray channel handshake。

当前 DreamerVLA 的链路大致是：

```text
RealEnvWorker rank 0
  slot 0 -> key 0:0 -> channel put/get
  slot 1 -> key 0:1 -> channel put/get
  slot 2 -> key 0:2 -> channel put/get

RolloutWorker
  逐个 key 等 env obs
  收集多个 slot 后做 batch policy forward
  再逐个 key 把 action/result 发回 env
```

也就是说，虽然 rollout 的模型前向可能是 batch 的，但 Ray channel 通信仍然按 slot 放大。slot 数越多、step 越多，Ray actor method call 和 channel event 越多，也越容易在某个 slot 上形成阻塞或放大控制面压力。

RLinf 的方案不是这样。RLinf 的 LIBERO 并行 env 主要藏在 EnvWorker 内部：

```text
EnvWorker rank 0
  内部管理多个 LIBERO subprocess env
  一次生成 batched observation
  一次接收 batched action
  内部 chunk_step / vector step 多个 env

RolloutWorker
  接收 batch
  做 batch policy forward
  返回 batch action
```

所以 RLinf 里 Ray 看到的是更粗粒度的 WorkerGroup 调用和 batch 级别数据流，而不是每个 env slot 都单独变成 Ray channel key。这也是为什么 RLinf 的同卡并行 LIBERO env 不容易因为 Ray GCS task event 被压垮。

本次应优先把 DreamerVLA 改向 RLinf 的思路：

1. 保留已经合入的 Ray task event 缓解，不要删掉 `enable_task_events=False` 和 `RAY_task_events_max_num_task_in_gcs` 设置。
2. 检查 `dreamervla/scheduler/channel.py` 里的 `_ChannelActor`，如果它仍然没有关闭 task events，也应给 ChannelActor options 加上 `enable_task_events=False`，因为 Channel 的 `put/get` 也是 Ray actor method call。
3. 更重要的是，逐步减少 per-slot Ray channel handshake，把 RealEnv/Rollout 的交互从 `rank:slot` 粒度改成 `rank` 粒度的 batch message。
4. 参考 RLinf 的 LIBERO EnvWorker / vector env / subprocess env 设计，让一个 EnvWorker 内部管理多个 env slot，对外只发 batched obs、接收 batched action。
5. RolloutWorker 应消费 batched obs，做一次 batch policy forward，然后返回 batched action/result，而不是每个 slot 单独阻塞等待。
6. Actor trajectory 收集也应尽量按 batch/shard 组织，避免每个 env slot 都形成独立 Ray channel 压力。

判断一个修改是否正确，不要只看 `WAIT` 日志是否减少，而要看：

* Ray actor 是否不再因为 GCS task event / SYSTEM_ERROR 死亡；
* global step 是否能稳定推进；
* 每个 global step 是否能完整完成 env -> rollout -> actor 的闭环；
* 增加 `envs_per_worker` 时，Ray task event 和 channel put/get 数量是否不再按 slot 线性爆炸；
* 代码结构是否向 RLinf 的 batched EnvWorker / RolloutWorker 数据流靠拢。

一句话总结：当前问题的根源不是某一行 `WAIT`，而是 DreamerVLA 把并行 env slot 暴露成了细粒度 Ray channel 通信单位；解决方案是保留 Ray task event 缓解补丁，同时把 RealEnv/Rollout/Actor 数据流改成 RLinf 风格的 batch/vector env 方案，从源头减少 Ray 控制面压力和 per-slot 阻塞风险。

---

## 2026-07-01 新方案复核补充：real env slot 进程隔离风险

截至 2026-07-01 的新方案已经把 Env/Rollout 主通信链路从 per-slot `rank:slot` 改成了 per-rank batch message，这个方向是对的，也能显著降低 Ray channel / GCS task event 压力。

但是请注意：新方案还不是完全等同于 RLinf 的 LIBERO vector env 方案。一个重要差异是：DreamerVLA 当前把 real env slot 放回 Ray worker 进程内跑了。

对应代码位置：

`/mnt/data/spoil/workspace/DreamerVLA/dreamervla/workers/env/trajectory_env_worker.py`

当前 real env 初始化逻辑类似：

```python
self.envs = [first_env] + [
    _build_env_from_cfg(self.env_cfg) for _ in range(self.num_slots - 1)
]
```

也就是说，一个 `RealEnvWorker` Ray actor 内部直接持有多个 env 对象。这样可以减少子进程数量和跨进程通信复杂度，但也意味着如果 LIBERO / MuJoCo / rendering 在进程内发生硬崩，整个 Ray worker actor 可能会一起死掉。

同时，新方案明确拒绝旧的 `spawn_env_slots` 配置，相关逻辑在同一个文件中：

```python
raw = self.env_cfg.get("spawn_env_slots", False)
enabled = str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
if enabled:
    raise ValueError(
        "env_cfg.spawn_env_slots has been removed from manual cotrain; "
        "EnvWorker slots now run in the Ray actor process."
    )
```

这和 RLinf 仍有差别：RLinf 的 LIBERO 并行环境通常藏在 EnvWorker 内部的 vector/subprocess env 中，Ray 只看到粗粒度 batch 交互，而单个 env 的 subprocess 仍有一定隔离性。DreamerVLA 当前方案虽然也减少了 Ray 通信粒度，但 real env 的崩溃隔离比 RLinf 弱。

因此，后续 Codex 继续实现或 debug 时需要区分两个层面：

1. **Ray 控制面压力问题**：新方案通过 per-rank batch message 和 `enable_task_events=False` 已经在正确方向上缓解。
2. **real env 进程内崩溃风险**：仍需要真实 LIBERO smoke test / long run 验证，尤其是 `envs_per_worker > 1` 时 MuJoCo、OSMesa/EGL、OpenGL context、内存和文件句柄是否稳定。

判断新方案是否真正稳定，不应只看单测通过，也不应只看 `WAIT` 日志减少。至少要验证：

* `manual_cotrain.envs_per_worker=1` 能完成一个完整 global step；
* 逐步提高到 `envs_per_worker=2/4/8` 后，Ray actor 不因为 real env 进程内崩溃而死亡；
* global step 能稳定推进，Actor/Rollout/Learner/Env 四组都完成闭环；
* 若 real env 仍会硬崩，应考虑恢复 RLinf 风格的 EnvWorker 内部 subprocess/vector env 隔离，而不是回到 per-slot Ray channel handshake。

---

## 2026-07-01 WoVR/RLinf 对照补充：group_size 不是流式训练触发条件

请注意：对照 RLinf 的 WoVR / Wan world-model 路线后，可以确认 `group_size=8` 在 RLinf/WoVR 中也不是“凑够 8 条 trajectory shard 就立刻训练”的触发条件。

WoVR 使用 RLinf embodied runner。训练时序在：

`/mnt/data/spoil/workspace/RLinf/rlinf/runners/embodied_runner.py`

核心顺序是：

```text
env.interact(...)
rollout.generate(...)
actor.recv_rollout_trajectories(...).wait()
rollout_handle.wait()
actor.compute_advantages_and_returns()
actor.run_training()
```

也就是说，Actor 可以在 Env/Rollout 运行期间开始接收 trajectory，但真正的 advantage 计算和 actor training 都发生在本轮 rollout 收集完成之后。`group_size=8` 是 GRPO / advantage 的分组单位，不是 streaming update 的触发阈值。

RLinf EnvWorker 的收集逻辑也说明了这一点：

`/mnt/data/spoil/workspace/RLinf/rlinf/workers/env/env_worker.py`

它在 `_run_interact_once` 中循环：

```text
for epoch in rollout_epoch:
    bootstrap obs
    for chunk_step_idx in n_train_chunk_steps:
        recv rollout result
        append ChunkStepResult
        env.chunk_step(...)
        send next obs

after all epochs:
    send_rollout_trajectories(..., actor_channel)
```

也就是说，它先把一个 rollout epoch/global step 需要的数据都组织成 `EmbodiedRolloutResult` / `Trajectory`，再发送给 Actor。不是每收到一个 group 就训练。

WoVR 的典型配置：

`/mnt/data/spoil/workspace/RLinf/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft_4567.yaml`

关键参数类似：

```yaml
algorithm:
  group_size: 8
  rollout_epoch: 16

env:
  train:
    total_num_envs: 64
    max_steps_per_rollout_epoch: 256

actor:
  global_batch_size: 8192
  micro_batch_size: 32
```

这里的 `group_size=8` 用于把 batch 中的 trajectories 按 8 个一组做 GRPO reward/advantage 计算。`actor.global_batch_size=8192` 才是 actor training 切 global batch / micro batch 的训练尺度。不要把 `group_size` 理解成“8 条来了就 optimizer step”。

还要注意 WoVR 和当前 DreamerVLA manual cotrain 的一个关键差异：

WoVR 的训练 env 是 Wan world model env：

`/mnt/data/spoil/workspace/RLinf/rlinf/envs/world_model/world_model_wan_env.py`

WanEnv 的 `chunk_step` 是 batch world-model inference：

```text
actions: [num_envs, chunk, action_dim]
pipe(..., batch_size=B)
reward_model predict batch rewards
return chunk rewards / dones
```

它不是慢速真实 LIBERO/MuJoCo RealEnv step，也不是 RealEnv + WMEnv 混在同一个 barrier 里训练。WoVR 的 eval 可以用 LIBERO real/sim env，但训练 loop 里的 env.train 是 Wan simulator。

因此，DreamerVLA 当前现象要这样理解：

* 当前代码“等整个 EnvGroup.interact 完成后再 actor training”的行为，与 RLinf/WoVR 的同步 PPO/GRPO 训练时序一致；
* `group_size=8` 不应该触发提前训练；
* 如果 DreamerVLA 把 RealEnv 和 WMEnv 放在同一个 global step barrier 内，那么 Actor 开始训练一定会等最慢的那一侧；
* 当前 WMEnv 先完成、RealEnv 还在采样时，阻塞 actor training 的就是 RealEnv 剩余数据；
* 这不是 WoVR 的典型训练形态，因为 WoVR 训练侧主要是 Wan world model simulator，不把慢 RealEnv 采样混入同一个必须等待的 rollout barrier。

如果后续希望 DreamerVLA 更接近 WoVR，有两个方向：

1. **WoVR-style 同步训练**：训练 loop 主要使用 WMEnv/WanEnv 作为 env.train，RealEnv 只用于 warmup、replay seed、validation/eval 或周期性校准，不让慢 RealEnv 阻塞每个 actor update。
2. **Hybrid Real+WM 训练**：可以保留 RealEnv + WMEnv 混合数据，但要接受同步 barrier 会被 RealEnv 拖慢；如果希望“WMEnv 先到先训”或“凑够 group_size 就训”，那已经不是当前 RLinf/WoVR 同步设计，需要重新设计 streaming actor update、policy version、advantage grouping、FSDP optimizer step、actor-to-rollout sync 和 replay/learner 同步语义。

一句话：WoVR 也是“收完整轮 rollout batch 后统一训练”，不是“8 条一组立刻训练”；DreamerVLA 当前等待 RealEnv 的行为符合同步设计，但它比 WoVR 慢的根本原因是把真实 RealEnv 采样混进了同一个 global step barrier。

---

## 2026-07-01 WM rollout 总条数预算补充：固定 target，不随卡数漂移

请按下面这个逻辑继续完成 DreamerVLA manual cotrain 的 WM rollout 预算改造。

这次改动的目标不是改模型训练逻辑，不是改 Actor 的 PPO/GRPO 更新方式，也不是改 env step 本身。目标只是在数据收集预算层面，把 WMEnv 的采样量从“每个 worker 固定 epoch”改成“每个 global step 固定总 trajectory 条数”。

旧逻辑是：

```text
WM 总条数 = wm_worker_count * envs_per_worker * wm_rollout_epoch
```

这里的 `wm_rollout_epoch` 实际含义是：每个 WM worker 里的每个 slot 采多少条 trajectory。这个设计会让 WM 总条数随卡数变化。例如同样的 `wm_rollout_epoch`，4 卡和 6 卡因为 `wm_worker_count` 不同，最终 WM 总采样量不同，不利于稳定对齐 WoVR 式固定 batch 预算。

新的目标逻辑是：

```text
WM 总目标条数 = manual_cotrain.wm_rollout_target_trajectories
默认目标：1024
```

runner 根据当前 `wm_worker_count` 和 `envs_per_worker` 自动把这个总目标分配到每个 WM worker 的 `rollout_epoch`。

计算方式：

```text
total_wm_worker_epochs = wm_rollout_target_trajectories / envs_per_worker
per_worker_epochs = total_wm_worker_epochs 平均分给 wm_worker_count 个 WM worker
```

如果不能整除，需要 fail fast：

```text
wm_rollout_target_trajectories % envs_per_worker == 0
```

并且必须保证每个 WM worker 至少拿到 1 个 rollout epoch：

```text
wm_rollout_target_trajectories / envs_per_worker >= wm_worker_count
```

例子：4 卡时通常是 1 个 RealEnv worker + 3 个 WMEnv worker。如果：

```yaml
manual_cotrain:
  envs_per_worker: 2
  wm_rollout_target_trajectories: 1024
```

那么：

```text
total_wm_worker_epochs = 1024 / 2 = 512
3 个 WM worker 分配为大约 [171, 171, 170]
最终 WM 条数 = (171 + 171 + 170) * 2 = 1024
```

RealEnv 仍然按时间预算调小，不要求和 WMEnv 条数相同：

```text
Real 条数 = real_worker_count * envs_per_worker * real_rollout_epoch
```

核心目标是：

* 每条 trajectory 的最大 step 长度固定，例如 `max_steps_per_rollout_epoch: 512`；
* `num_action_chunks=8` 时，每条 trajectory 是 `512 / 8 = 64 chunk_step`；
* WMEnv 每个 global step 总共采集约 `wm_rollout_target_trajectories=1024` 条；
* RealEnv 条数可以少一些，根据 wall-clock 时间调，不要让 RealEnv 成为每个 global step 的主要瓶颈；
* Actor 仍然等完整 EnvGroup 数据后统一 `compute_advantages_and_returns()` 和 `run_training()`，不要改成凑够 `group_size=8` 就 streaming train。

推荐配置语义：

```yaml
manual_cotrain:
  max_steps_per_rollout_epoch: 512
  wm_rollout_multiplier: 1
  num_action_chunks: 8

  real_rollout_epoch: 4

  # fallback only，当 target 未设置时才直接使用
  wm_rollout_epoch: ${manual_cotrain.rollout_epoch}

  # 新主逻辑：固定每轮 WM 总 trajectory 条数
  wm_rollout_target_trajectories: 1024
```

实现要求：

1. `ManualCotrainRayRunner` 需要提供 `_wm_rollout_target_trajectories()` 和 `_wm_rollout_epochs_by_worker(worker_count)`。
2. 如果 `wm_rollout_target_trajectories` 存在，实际 WM epoch 分配必须以 target 为准，而不是直接使用 `wm_rollout_epoch`。
3. `WMEnvWorker` 启动后，runner 必须能给每个 WM worker 设置自己的 rollout_epoch。可以通过 `execute_on(rank).configure_rollout_epoch(epoch)` 实现。
4. `BaseTrajectoryEnvWorker.configure_rollout_epoch()` 只更新该 worker 的 `self.rollout_epoch`，不改 step 长度、不改 slot 数、不改 action chunk 逻辑。
5. `_configured_expected_trajectory_shards()` 必须用新逻辑计算：

```text
expected_shards =
  real_workers * envs_per_worker * real_rollout_epoch
  + sum(wm_rollout_epochs_by_worker) * envs_per_worker
```

6. `dreamervla/config.py` 的 manual cotrain group geometry 校验必须同步新语义：

```text
real_trajectory_count = real_workers * envs_per_worker * real_rollout_epoch

if wm_rollout_target_trajectories is set:
    wm_trajectory_count = wm_rollout_target_trajectories
else:
    wm_trajectory_count = wm_workers * envs_per_worker * wm_rollout_epoch

logical_trajectory_count = real_trajectory_count + wm_trajectory_count
logical_trajectory_count % group_size == 0
```

7. 配置校验还必须覆盖：

```text
wm_rollout_target_trajectories > 0
wm_rollout_target_trajectories % envs_per_worker == 0
wm_rollout_target_trajectories / envs_per_worker >= wm_worker_count
```

8. 单测必须覆盖至少这些场景：

```text
4 卡：wm_workers=3, envs_per_worker=2, target=1024 -> epochs [171,171,170] 或同等分配，总条数 1024
6 卡：wm_workers=5, envs_per_worker=2, target=1024 -> 总条数仍是 1024
target 不能被 envs_per_worker 整除 -> 抛错
target 太小，无法让每个 WM worker 至少 1 epoch -> 抛错
expected_shards 使用 target 逻辑，不随 wm_worker_count 线性漂移
configure_rollout_epoch 只影响指定 WM worker
```

需要特别避免的错误：

* 不要把 `wm_rollout_target_trajectories` 理解成 step 数；它是 trajectory shard 条数。
* 不要把 `max_steps_per_rollout_epoch` 理解成条数；它是每条 trajectory 的最大底层 env action step 长度。
* 不要因为 `group_size=8` 就提前训练；`group_size` 仍然只是 GRPO advantage 分组单位。
* 不要让 `wm_rollout_target_trajectories` 和 `wm_rollout_epoch` 同时生效导致重复放大。target 存在时，target 是主逻辑，`wm_rollout_epoch` 只是 fallback。

一句话：这次要完成的是“WM 每个 global step 固定总采样条数”的预算逻辑，使 WM 总条数稳定为 1024 左右，不再因为 GPU/WM worker 数变化而漂移；RealEnv 条数单独按时间预算控制，少一点可以接受，重点是不要拖慢每个 global step。

---

## 2026-07-01 配置原则修正：对齐 WoVR，但 RealEnv/WMEnv 分开预算

更准确的目标不是发明一套和 WoVR/RLinf 不同的训练逻辑，而是继续对齐 WoVR 的核心思想：

```text
固定 rollout batch 预算
固定每条 trajectory 的最大 step 长度
收完整轮 rollout 后统一计算 advantage 和训练
group_size 只作为 GRPO 分组单位
```

DreamerVLA 和 WoVR 的差异在于：DreamerVLA manual cotrain 同时有两个 env 数据源：

```text
RealEnv: 真实 LIBERO/MuJoCo 采样，慢，wall-clock 成本高
WMEnv: world model imagined rollout，快，主要提供大规模训练 batch
```

所以配置上不能只照搬 WoVR 的单一 `env.train` 预算字段，而要把 RealEnv 和 WMEnv 的预算拆开：

```text
WMEnv 预算：对齐 WoVR 的主训练 batch 预算，固定每个 global step 的总 trajectory 数。
RealEnv 预算：按 wall-clock 调小，作为真实数据/校准数据来源，不要拖慢每个 global step。
```

推荐的语义是：

```yaml
manual_cotrain:
  # 共同的每条 trajectory 最大底层 action step 长度。
  # 如果未来 Real/WM 需要不同长度，再拆成 real_max_steps_per_rollout_epoch
  # 和 wm_max_steps_per_rollout_epoch。
  max_steps_per_rollout_epoch: 512
  num_action_chunks: 8

  # RealEnv 慢，按时间预算控制。
  real_rollout_epoch: 4

  # WMEnv 是主训练 batch 来源，按总条数控制，不随 WM worker 数漂移。
  wm_rollout_target_trajectories: 1024
  wm_rollout_multiplier: 1

  # fallback only: 只有未设置 wm_rollout_target_trajectories 时才使用。
  wm_rollout_epoch: ${manual_cotrain.rollout_epoch}
```

在这个配置下：

```text
每条 trajectory 长度 = 512 / 8 = 64 chunk_step

WMEnv 每个 global step 总条数 = 1024 trajectory shards
RealEnv 每个 global step 总条数 = real_workers * envs_per_worker * real_rollout_epoch
```

这才是和 WoVR 对齐后的 DreamerVLA 版本：训练仍然是同步 PPO/GRPO，不做 8 条一到就训练；但因为 DreamerVLA 有 RealEnv + WMEnv 双 env 来源，所以 WMEnv 用固定总 batch 预算，RealEnv 用独立的小预算。

实现和测试时请坚持这个边界：

* 不要把 RealEnv 和 WMEnv 强行配置成相同条数；RealEnv 慢，少一些是合理的。
* 不要让 WM 总条数随卡数变化；`wm_rollout_target_trajectories` 应该保证 4 卡、6 卡等不同 `wm_worker_count` 下 WM 总条数仍稳定。
* 不要改 Actor 的训练触发逻辑；仍然等完整 EnvGroup 数据后统一训练。
* 不要把 `wm_rollout_epoch` 和 `wm_rollout_target_trajectories` 同时乘起来；target 存在时 target 是主逻辑。
* 如果将来发现 RealEnv 的每条 trajectory 长度也应该和 WM 不同，再新增 `real_max_steps_per_rollout_epoch` / `wm_max_steps_per_rollout_epoch`，不要复用 multiplier 做隐式语义。

一句话：更好的方案是“WoVR-style 同步训练预算 + DreamerVLA 双 env 独立预算”。WMEnv 对齐 WoVR 的固定大 batch，RealEnv 用较小真实采样预算补充和校准，不让 RealEnv 成为主训练吞吐瓶颈。
