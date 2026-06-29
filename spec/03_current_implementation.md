# Current Implementation

状态：current

本文记录当前代码已落地的主线事实。目标设计仍以 [`99_manual_notes.md`](99_manual_notes.md) 为最高
优先级；当本文件与代码不一致时，先重新检查代码和测试，再更新本文件。

## Mainline Route

当前 OpenVLA-OFT cold-start 主线：

```text
collect rollouts
-> seed replay
-> warm up world model + classifier
-> manual Ray cotrain
-> real LIBERO eval
```

入口有两层：

- 训练入口：`python -m dreamervla.train experiment=<name> task=<suite>`。
- pipeline 入口：`python -m dreamervla.launchers.coldstart_warmup_cotrain mode=ray`，由
  `scripts/e2e_coldstart_warmup_cotrain_{ray,noray}.sh` 调用。

`configs/scripts/coldstart_warmup_cotrain.yaml` 默认 `cotrain_engine=sync`；设为 `async` 时走
Ray 主线：

```text
cotrain_engine=async
cotrain_async_experiment=manual_cotrain_ray_oft_backbone_latent
```

因此 Ray async 主线是 `ManualCotrainRayRunner`，不再是旧 `OnlineCotrainRayRunner`。旧 runner 仅作
显式 legacy/optional 实验保留。

代码定位：

- launcher: `dreamervla/launchers/coldstart_warmup_cotrain.py`
- manual runner: `dreamervla/runners/manual_cotrain_ray_runner.py`
- config: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
- tiny config: `configs/experiment/manual_cotrain_ray_tiny.yaml`
- scheduler: `dreamervla/scheduler/worker_group.py`、`dreamervla/scheduler/channel.py`
- placement: `dreamervla/workers/cotrain/placement.py`
- messages: `dreamervla/workers/cotrain/messages.py`

## Implemented Group Topology

`ManualCotrainRayRunner` 创建以下 Ray group：

| Ray group | Worker | 当前职责 |
| --- | --- | --- |
| `LearnerGroup` | `LearnerWorker(mode=wm_classifier_only)` | 只更新 world model 和 classifier/reward model。 |
| `ActorGroup` | `EmbodiedFSDPActor` | 训练 VLA policy：PPO loss、backward、optimizer step、FSDP state export。 |
| `RolloutGroup` | `MultiStepRolloutWorker` | `eval/no_grad` policy copy，生成 action chunk、old logprob 和 `forward_inputs`。 |
| `RealEnvGroup` | `RealEnvWorker` | 执行真实 LIBERO，按 chunk 组装 trajectory shard，写 replay sidecar。 |
| `WMEnvGroup` | `WMEnvWorker` | 可选；执行 latent WMEnv，从 LearnerGroup 同步 WM/cls。 |
| `ReplayGroup` | `ReplayWorker` | 可选；WM/cls 训练与 WMEnv bootstrap 数据，不是 Actor PPO 通道。 |

`RealEnvGroup`/`WMEnvGroup` 是概念上 `EnvGroup` 的两个实现分支。

## Implemented Placement

`build_manual_cotrain_placement(ngpu)` 对任意 `ngpu >= 0` 生成拓扑（主线验收覆盖 0–5 卡）：

```text
ngpu=0:
  real_env + rollout + actor + learner 全部 CPU，actor_fsdp_strategy="none"

ngpu=1:
  GPU0: RealEnvWorker + RolloutWorker + ActorGroup rank0 + LearnerWorker   (fsdp)

ngpu=N, N>1:
  GPU0:           RealEnvWorker + RolloutWorker + LearnerWorker
  GPU1..GPU(N-1): WMEnvWorker + ActorGroup rank
  RolloutWorker:  每张可见 GPU 各一份 (GPU0..GPU(N-1))
```

Actor rank 使用 GPU1..N-1；只有一张 GPU 时 actor 放 GPU0。`ngpu=0` 时 actor FSDP strategy 为
`none`。

代码定位：`dreamervla/workers/cotrain/placement.py`，测试
`tests/unit_tests/test_manual_cotrain_placement.py`。

## Implemented Global Step

当前 `_run_global_step(groups, global_step)` 顺序：

```text
1.  set_global_step              actor / rollout / real_env / (wm_env)
2.  actor_to_rollout_sync        if step % sync_every == 0:
                                   actor.sync_model_to_rollout("policy", step)
                                   rollout.sync_model_from_actor("policy")
                                   (optional) replay.set_policy_version(step)
3.  env_interact + generate      real_env.interact(env, rollout, actor)
                                   (optional) wm_env.interact(...)
                                   rollout.generate(env, rollout, envs_per_worker)
                                   async actor trajectory receivers + rollout-failure guard
4.  actor_recv_trajectories      env_channel.put(StopMsg("global_step_complete"))
                                   收齐 trajectory shard
5.  actor.compute_advantages_and_returns()
6.  actor.run_training()
7.  learner_update               if step % learner_update_step == 0:
                                   learner.update("cotrain", 1)
                                   learner.sync_weights("world_model"/"classifier", step)
                                   learner.state_dicts() ->
                                   wm_env.load_component_states({world_model, classifier}, step)
8.  checkpoint_and_metrics       _maybe_save_manual_checkpoint(...) + metrics 汇总
```

Env/Rollout 通过命名 channel 并发；runner 等 EnvGroup 完成的同时监视 RolloutGroup 早退失败。
配置 accessor 默认值：`sync_every=1`、`learner_update_step=1`、`global_steps=1`、`rollout_epoch=1`、
`num_action_chunks=1`、`envs_per_worker=1`、`checkpoint_every=0`（关闭）。

## Implemented Data Path

manual path 使用 typed cotrain messages（见 [`05_cotrain_data_contracts.md`](05_cotrain_data_contracts.md)）：

- `ObservationMsg`：EnvGroup -> RolloutGroup。
- `RolloutResultMsg`：RolloutGroup -> EnvGroup。
- `TrajectoryShard`：EnvGroup -> ActorGroup（chunk-level，leading `[1, 1, ...]`）。
- `TrajectoryBatch`：ActorGroup collation 输出（`collate_trajectory_shards` 沿 `dim=1` 拼接、pad
  到 max_steps）。
- `StopMsg`：channel 控制/收尾消息。

真实 env obs 不必自带 `obs_embedding`：`MultiStepRolloutWorker` 用 `OFTRolloutBundle` 把 image obs
编码成 `obs_embedding`/`lang_emb`，放进 `forward_inputs`，EnvWorker 再写入 replay transition
sidecar。`env.real.cfg.action_postprocess=openvla_oft` 时，只有 env step 收到 postprocess 后的
gripper 动作；replay 和 ActorGroup 仍使用原始 policy action chunk。

WMEnv bootstrap 在 replay 可用时采样初始 `obs_embedding`、`lang_emb`、`proprio`；缺失时回落到 env
配置的 reset 行为。

## Configs Snapshot

`configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`（Ray async 主线）关键值：
`ngpu=1`、`global_steps=1`、`learner_update_step=1`、`sync_every=1`、`rollout_epoch=${algorithm.rollout_epoch}`(=16)、
`max_steps_per_rollout_epoch=256`、`wm_rollout_multiplier=4`、
`num_action_chunks=${task.openvla_oft.input_tokens.chunk_size}`、`envs_per_worker=8`、
`algorithm.group_size=8`；`learner.init_ckpt={path: ${init.warmup_ckpt_path}, components: [world_model, classifier]}`。

`configs/experiment/manual_cotrain_ray_tiny.yaml`（CPU smoke）关键值：`ngpu=0`、`global_steps=1`、
`learner_update_step=999999`、`sync_every=1`、`rollout_epoch=1`、`max_steps_per_rollout_epoch=2`、
`num_action_chunks=2`、`envs_per_worker=2`、`requires_bootstrap_value=false`、
`wm_env_write_replay=false`。

## Current Validation State

Unit 覆盖（见 [`07_validation_matrix.md`](07_validation_matrix.md)）：

- public runner export 与 route composition
- manual placement（0–5 GPU）
- cotrain messages 与 chunk-level collation
- rollout worker hidden extraction、OFT encoding、patch sync
- trajectory env worker chunk-level stepping、sidecar attachment、EGL spawn recovery、WM sync
- actor PPO shape handling 与 FSDP-safe state export
- learner `wm_classifier_only` 行为与 precision validation
- cold-start launcher 命令构造与 warmup checkpoint bridge
- manual cotrain config validation（FSDP strategy、chunk divisibility）

剩余高价值验证是运行级：

- 目标 GPU/LIBERO 机器上完整跑通 `scripts/e2e_coldstart_warmup_cotrain_ray.sh cotrain_engine=async`。
- Ray placement、EGL/MuJoCo、OpenVLA-OFT checkpoint 加载、长时间 replay 写入的稳定性。
- 训练后 VLA 的最终 real LIBERO eval 指标。
