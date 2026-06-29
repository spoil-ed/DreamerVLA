# Ray Implementation

状态：目标 Ray/worker 拓扑说明

这份文档说明如何用 Ray worker group 实现 [`01_complete_loop.md`](01_complete_loop.md) 的主线。
当前 manual route 已落地为 `ManualCotrainRayRunner`；实现事实见
[`03_current_implementation.md`](03_current_implementation.md)，与 RLinf 的对齐见
[`04_rlinf_alignment.md`](04_rlinf_alignment.md)。目标判断仍以
[`99_manual_notes.md`](99_manual_notes.md) 为准。

## 1. Scheduler Primitives

主线不让 driver 手动串联每一步，而是建立在两个调度原语上：

- `WorkerGroup`（`dreamervla/scheduler/worker_group.py`）：把一类 worker 包成一组 Ray actor。
  `launch(cluster, placement, name, env_vars)` 按 placement 给每个 rank 起一个 Ray actor 并调
  `init`；`group.<method>(...)` 把方法广播到该 group 的全部 rank，返回一个
  `WorkerGroupFuncResult`，`.wait()` 才真正 `ray.get`；`execute_on(*ranks)` 把下一次调用限制到
  指定 rank（例如只在 actor rank0 导出 state）。GPU 绑定通过 `CUDA_VISIBLE_DEVICES` 和
  `RANK/LOCAL_RANK/WORLD_SIZE` env 注入。
- `Channel`（`dreamervla/scheduler/channel.py`）：命名 Ray actor，提供 `put`/`get`，支持按
  `key` 分流的多条 FIFO 队列（默认 `key="default"`）。`create(name)` 在 namespace `DreamerVLA`
  下建 detached actor，`connect(name)` 连接已有 channel。`StopMsg` 只是一个普通消息，靠它在队列
  里传递 shutdown 信号。

## 2. Target Worker Graph

目标 Ray 图是四个角色 group（外加可选 replay）：

```text
LearnerGroup   -> LearnerWorker
ActorGroup     -> EmbodiedFSDPActor rank(s)
RolloutGroup   -> MultiStepRolloutWorker rank(s)
EnvGroup       -> RealEnvWorker (real_env) / WMEnvWorker (wm_env)
(ReplayGroup)  -> ReplayWorker        # 可选，WM/cls 训练与 WMEnv bootstrap 数据
```

职责：

- `LearnerWorker`：非 FSDP，训练 world model 和 classifier/reward model。
- `EmbodiedFSDPActor`：FSDP，训练 VLA，处理 PPO loss、backward、optimizer、FSDP 通信。
- `MultiStepRolloutWorker`：非 FSDP，`eval/no_grad` 推理副本，只负责采样。
- `RealEnvWorker`：真实 LIBERO 环境 worker。
- `WMEnvWorker`：latent world-model env worker，内部持有 world model 和 classifier/reward model。

## 3. Placement

`build_manual_cotrain_placement(ngpu)` 对任意 `ngpu >= 0` 生成拓扑（主线验收覆盖 0–5 卡）：

```text
ngpu=0:  real_env + rollout + actor + learner 全部 CPU，actor_fsdp_strategy="none"

ngpu=1:  GPU0: RealEnvWorker + RolloutWorker + ActorGroup rank0 + LearnerWorker  (fsdp)

ngpu=N (N>1):
  GPU0:            RealEnvWorker + RolloutWorker + LearnerWorker
  GPU1..GPU(N-1):  WMEnvWorker + ActorGroup rank
  RolloutWorker:   每张可见 GPU 各一份 (GPU0..GPU(N-1))
```

Actor rank 使用 GPU1..N-1；只有一张 GPU 时 actor 也放 GPU0。这只是默认 profile，不应硬编码：
实际 GPU 数、每卡 env slots、Actor world size 和是否启用真实环境卡都来自 Hydra。

## 4. Channels

Runner 建立三条命名 channel（名字带 `uuid` 后缀，避免多 run 撞名）：

```text
env_channel      manual-cotrain-env-<uuid>       承载 StopMsg 等 env 控制信号
rollout_channel  manual-cotrain-rollout-<uuid>   ObservationMsg <-> RolloutResultMsg
actor_channel    manual-cotrain-actor-<uuid>     TrajectoryShard 流向 ActorGroup
```

数据流（按 `key = "<env_rank>:<slot_id>"` 分流）：

```text
EnvWorker --ObservationMsg-->  rollout_channel  --> RolloutWorker.generate
RolloutWorker --RolloutResultMsg--> rollout_channel --> EnvWorker.apply_rollout_result
EnvWorker --TrajectoryShard--> actor_channel --> ActorGroup receivers
Runner    --StopMsg--> env_channel  --> EnvWorker 收尾
```

消息结构由 [`05_cotrain_data_contracts.md`](05_cotrain_data_contracts.md) 固定。

## 5. Global Step Orchestration

`ManualCotrainRayRunner._run_global_step(groups, global_step)` 的顺序（每步操作通过
`WorkerGroupFuncResult.wait()` 串行成屏障）：

1. `set_global_step`：actor / rollout / real_env /（可选 wm_env）都设置当前 step/version。
2. `actor_to_rollout_sync`（当 `global_step % sync_every == 0`）：
   `actor.sync_model_to_rollout("policy", global_step)` + `rollout.sync_model_from_actor("policy")`
   +（可选）`replay.set_policy_version(global_step)`。
3. `env_interact_and_rollout_generate`：`real_env.interact(env, rollout, actor)`、可选
   `wm_env.interact(...)` 与 `rollout.generate(env, rollout, envs_per_worker)` 并发启动；同时启动
   actor trajectory 异步接收，并在等待 EnvGroup 完成时监视 RolloutGroup 早退失败。
4. `actor_recv_trajectories`：向 env_channel 发 `StopMsg(reason="global_step_complete")`，等
   rollout 结束，收齐 actor trajectory shard。
5. `actor.compute_advantages_and_returns()`。
6. `actor.run_training()`。
7. `learner_update_wm_classifier`（当 `global_step % learner_update_step == 0`）：
   `learner.update("cotrain", 1)`，`learner.sync_weights("world_model"/"classifier", global_step)`，
   再 `learner.state_dicts()` 取出状态，调
   `wm_env.load_component_states({"world_model":..., "classifier":...}, global_step)` 同步给 WMEnv。
8. `checkpoint_and_metrics`：`_maybe_save_manual_checkpoint(...)`，并汇总 metrics。

## 6. EnvWorker

`BaseTrajectoryEnvWorker`（`role_name="env"`）是接口基类，`interact(env_ch, rollout_ch, actor_ch)`
做 rollout-epoch / chunk-step 两层循环：每个 rollout epoch 先 `bootstrap_obs()` 给每个 slot 发
`ObservationMsg`；当某 slot 还没到 `target_chunk_steps`，就从 rollout_channel 按 key 取
`RolloutResultMsg`，`apply_rollout_result` 执行该 chunk 并产出一个 `TrajectoryShard` 推进 actor
channel，然后把新 obs 发回 rollout；epoch 末做收尾和一次 final bootstrap。

`RealEnvWorker`（`role_name="real_env"`）：构建真实 LIBERO env，用真实 image/state 作 obs，并把
rollout 的 `hidden -> obs_embedding`、`lang_emb` 和 version sidecar 写进 replay transition。启用
EGL/MuJoCo 时，每个 slot 跑在 spawn 子进程里，并有子进程死亡后的 respawn 恢复。

`WMEnvWorker`（`role_name="wm_env"`）：加载 world model 和 classifier/reward model，从 replay
bootstrap 初始 latent，后续在 latent state 中 step，并通过
`load_world_model_state`/`load_classifier_state`/`load_component_states` 从 LearnerGroup 同步权重。

每个 EnvWorker 可有多个子 env slots；子 env 接 action、返回 state/image/latent obs，父 EnvWorker
负责 batch 管理和 trajectory 聚合。

## 7. RolloutWorker

`MultiStepRolloutWorker` 持有普通 `eval/no_grad` 推理副本（非 FSDP）。`generate(...)` 从 channel
取 `ObservationMsg`，把 obs 编码成 `hidden`（OFT 路径用 `OFTRolloutBundle`），policy 以
`mode="sample"` 输出 action chunk 和 logprob，返回 `RolloutResultMsg`。它必须把训练所需的
action/token/input 放进 `forward_inputs`（至少 `hidden` + `action`，可选 `lang_emb`、
`action_token_ids`、`input_ids`、`attention_mask`、`hidden_states`），因为 ActorGroup 之后要用这些
重算 logprob。`sync_model_from_actor("policy")` 通过 `PatchWeightSyncer.pull` 拉取 actor patch。

## 8. ActorGroup

ActorGroup 用 `EmbodiedFSDPActor`。每个 rank：

1. `load_trajectory_shards(...)` 收 shard 并 collate 成 `TrajectoryBatch`。
2. `compute_advantages_and_returns()`：return 对 time + chunk trailing 维求和得 `[B]`，再 GRPO
   group-relative 归一化成 advantage。
3. `run_training()`：用 `forward_inputs` 重跑当前 Actor 得 `new_logprobs`，做 PPO clipped loss
   （`clip_ratio_low`/`clip_ratio_high`/`clip_log_ratio` + entropy），经 FSDP backward / optimizer
   更新 VLA。强制 `batch.actions.ndim == 4`，不展平 chunk。
4. `state_dict()` / `sync_model_to_rollout(...)`：用 `FullStateDictConfig(rank0_only)` 让所有 rank
   一起进入 state-dict collective，只有 rank0 发布 patch。

value-based PPO/GAE（`prev_values`、`returns`、final bootstrap）是未来可选分支；当前主线是
trajectory 级别 reward，不需要它们。

## 9. LearnerGroup

LearnerGroup 非 FSDP，只负责 world model 和 classifier/reward model。它不训练 VLA，也不接管
ActorGroup 的 PPO batch。它按 `learner_update_step` 从配置指定数据流训练 WM/cls，再把新版本同步
给 WMEnvWorker。主线里没有额外 verifier：WMEnvWorker 的 reward model 就是这个 classifier。

## 10. Sync

- **ActorGroup 内部**：FSDP/NCCL 管理参数 shard、梯度同步和 optimizer state，不手动复制 rank。
- **ActorGroup -> RolloutGroup**：rollout 前 patch sync。rollout 端本地已有 HF VLA 副本，只 apply
  patch 到 `state_dict`；版本跳变过大时回落到全量 load。
- **LearnerGroup -> WMEnvWorker**：LearnerGroup 更新后，runner 显式把 WM/cls state 拷给 WMEnv。

## 11. Current Status

当前代码已有目标 route 和 worker：

```text
ManualCotrainRayRunner
  LearnerGroup -> LearnerWorker(mode=wm_classifier_only)
  ActorGroup   -> EmbodiedFSDPActor
  RolloutGroup -> MultiStepRolloutWorker
  RealEnvGroup -> RealEnvWorker
  WMEnvGroup   -> WMEnvWorker        # 可选
  ReplayGroup  -> ReplayWorker       # 可选
```

已落地：四 group 分离、chunk-level trajectory shard 直达 ActorGroup、Actor->Rollout patch sync
（含 FSDP full-state export）、Learner->WMEnv 显式 state 同步、0..N placement。剩余主要是运行级
验证：目标 GPU/LIBERO 机器上跑通 `cotrain_engine=async`，并验证 EGL/MuJoCo placement、
OpenVLA-OFT checkpoint 加载、长时间 replay 写入和最终 real LIBERO eval。验收清单与命令见
[`07_validation_matrix.md`](07_validation_matrix.md)。
