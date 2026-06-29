# Sync, Checkpoint, And Metrics

状态：current

本文固定 manual cotrain route 的同步、checkpoint 和 metrics 契约。

## Synchronization

三类相互独立的同步层：

| Sync | Owner | 机制 | 目的 |
| --- | --- | --- | --- |
| ActorGroup 内部 | `EmbodiedFSDPActor` / PyTorch FSDP | FSDP/NCCL | shard 参数、同步梯度、更新 optimizer state |
| ActorGroup -> RolloutGroup | `PatchWeightSyncer` | patch / state-dict delta | rollout 前更新推理副本 |
| LearnerGroup -> WMEnvWorker | runner 显式 load 调用 | state dict copy | 更新 WMEnv 的 world model 和 classifier/reward model |

**Actor -> Rollout** 不是 Actor 内部 FSDP 同步。Actor 导出 full state 时，所有 actor rank 必须一起
进入 state-dict collective（`FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`），只有
rank0 调 `sync_model_to_rollout("policy", version)` 发布 patch。`PatchWeightSyncer.push` 把当前
state 与上一份 full state 求 delta，存 `patch`、`full` 和 `meta`（含 `latest_version`、
`patch_keys`）。`pull` 时若本地版本正好落后一版就 apply patch，否则回落到全量 load。Rollout 端
只把 patch apply 到本地 HF policy `state_dict`，自身始终 `eval/no_grad`、非 FSDP。

**Learner -> WMEnv**：runner 调 `learner.sync_weights("world_model"/"classifier", step)` 发布版本，
再取 `learner.state_dicts()` 把 `{world_model, classifier}` 通过
`wm_env.load_component_states(..., step)` 显式拷给 WMEnvWorker。

## Warmup Checkpoint Bridge

sync warmup pipeline 写出拆分的组件 checkpoint：

```text
${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt
```

async manual cotrain 时，launcher 把它们合并成：

```text
${RUN_ROOT}/cotrain/ckpt/ray_async_init.ckpt
```

合并后的 payload：

```text
{
  "state_dicts": {
    "world_model": ...,
    "classifier": ...,
  },
  "classifier_threshold": optional float,
}
```

`ManualCotrainRayRunner` 通过 `init.warmup_ckpt_path` 加载它。config 里
`learner.init_ckpt={path: ${init.warmup_ckpt_path}, components: [world_model, classifier]}`，runner
在 worker 启动前按 component 解析，只把 `world_model`/`classifier` 传给 LearnerGroup。

## Manual Checkpoints

manual cotrain checkpoint 可选，由 `manual_cotrain.checkpoint_every` 控制（`0` 关闭）。启用时 runner
写入：

```text
${training.out_dir}/checkpoints/manual_cotrain_step_<N>/manual_cotrain.ckpt
${training.out_dir}/checkpoints/manual_cotrain_step_<N>/manual_cotrain_manifest.json
```

payload：

```text
{
  global_step,
  metrics,
  state_dicts: { policy, world_model, classifier },
  replay: optional,
}
```

manifest（`schema_version=1`）记录 `global_step`、versions（`actor_policy_version`、
`rollout_policy_version`、`wm_version`、`classifier_version`）、components、replay 信息和 metrics
key。Actor state 导出必须 FSDP-safe：所有 actor rank 参与，runner 保留第一份非空 mapping。

## Metrics Namespaces

manual cotrain 保持仓库统一 namespace（通过 `BaseRunner.log_metrics` 流转）：

| Prefix | 示例 |
| --- | --- |
| `env/` | `chunk_steps`、`physical_steps`、`trajectory_shards`、`episodes_completed`、`env_crashes` |
| `rollout/` | `generated` |
| `actor/` | `trajectory_count`、`return_mean`、`advantage_std`、`ppo_updates`、`loss`、`ratio_mean` |
| `train/` | LearnerWorker 发出的 WM/classifier loss |
| `sync/` | `policy_version`、world-model/classifier version 与 sync 计时 |
| `replay_buffer/` | `size`、`transitions` |
| `time/` | 可得的计时指标 |
