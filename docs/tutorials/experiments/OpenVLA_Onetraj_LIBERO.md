# OpenVLA-OFT One-Trajectory LIBERO Cotrain

本文给出当前主线 OpenVLA-OFT one-trajectory cotrain 的完整运行方案。实现和
配置以 [manual notes](../../../spec/99_manual_notes.md)、
[complete loop](../../../spec/04_complete_loop.md) 和
[route registry](../../../spec/06_routes.md) 为准。

## 关键实验命令速查

以下命令均在 DreamerVLA 根目录执行；8 卡默认使用 GPU `0-7`：

```bash
export DVLA_DATA_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 官方数据 WM / classifier 上限训练（两个独立实验）
bash scripts/experiments/world_model_training/train.sh
bash scripts/experiments/classifier_training/train.sh

# 64-step WM 时耗诊断：前 32 步分段 profile，后 32 步观察实际吞吐
bash scripts/experiments/world_model_training/profile.sh

# 恢复中断的 WM 训练；脚本自动选择最新 warmup_progress
WORLD_MODEL_RESUME=true \
WORLD_MODEL_RUN_ROOT=/path/to/world_model/run \
  bash scripts/experiments/world_model_training/train.sh

# 冻结任意选定的兼容 WM/CLS checkpoint，启动 8 卡 Ray policy-only cotrain
WORLD_MODEL_CKPT=/path/to/world_model/run-or-checkpoint \
CLASSIFIER_CKPT=/path/to/classifier/run-or-checkpoint \
  bash scripts/e2e_frozen_model_cotrain.sh

# 恢复冻结 cotrain
WORLD_MODEL_CKPT=/path/to/world_model/run-or-checkpoint \
CLASSIFIER_CKPT=/path/to/classifier/run-or-checkpoint \
COTRAIN_RESUME_CKPT=/path/to/manual_cotrain.ckpt \
  bash scripts/e2e_frozen_model_cotrain.sh

# 正式主线：collection -> warmup -> 8 卡 Ray async online cotrain
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
```

冻结 cotrain 的 `WORLD_MODEL_CKPT` 可以直接指向任意兼容 WM checkpoint，
包括 `warmup_topk` 或 `warmup_progress`。传 run 目录时，launcher 依次选择当前
最低-loss top-k、最终 checkpoint、最新 progress，不要求训练完成。额外实验
参数继续使用 Hydra `key=value` override。`CLASSIFIER_CKPT` 同样可直接指向
`best_window_*.ckpt`、`final.ckpt` 或 `latest.ckpt`；传 run 目录时依次选择最高
window-F1、`final.ckpt`、`latest.ckpt`。通用 final/latest checkpoint 的模型状态
从 `state_dicts.model` 读取，阈值优先沿用同目录最高-F1 checkpoint；尚无校准
checkpoint 时显式采用 `0.5`，也可用
`algorithm.lumos.classifier_threshold=<value>` 覆盖。

## 1. 主线路由

完整流程是：

```text
cold-start collection
  -> seed offline replay
  -> DDP warmup world model + success classifier
  -> consolidate Ray init checkpoint
  -> manual Ray async online cotrain
  -> real LIBERO evaluation
```

各阶段的当前入口如下：

| 阶段 | Hydra experiment | Runner |
| --- | --- | --- |
| Ray collection | `collect_rollouts_ray` | `ColdStartRayCollectRunner` |
| no-Ray collection | `collect_rollouts_onetraj` | `CollectRolloutsRunner` |
| DDP warmup / sync cotrain | `openvla_onetraj_libero_cotrain_noray` | `OnlineCotrainPipelineRunner` |
| async online cotrain | `openvla_onetraj_libero_cotrain_ray` | `ManualCotrainRayRunner` |
| evaluation | `eval_libero_vla` | `EmbodiedEvalRunner` |

`openvla_onetraj_libero_cotrain_ray_base` 是共享配置基座，仍以
`OnlineCotrainRayRunner` 为 target。它不是独立主线 experiment，不要直接启动。
sync no-Ray 和 manual Ray async 都是受支持主线；8 卡重训练推荐使用 async wrapper。

## 2. Config 是实验核心

shell 只负责环境和启动。模型、数据、优化器、batch、学习率、预算、placement、
checkpoint 和评估都由 Hydra 配置决定。

| 配置 | 负责内容 |
| --- | --- |
| [coldstart_warmup_cotrain.yaml](../../../configs/scripts/coldstart_warmup_cotrain.yaml) | launcher、profile、阶段拆分、8 卡规模和下游 override |
| [openvla_onetraj_libero_cotrain_noray.yaml](../../../configs/dreamervla/openvla_onetraj_libero_cotrain_noray.yaml) | warmup/sync 模型、优化器、replay 和 online 配置 |
| [openvla_onetraj_libero_cotrain_ray.yaml](../../../configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml) | manual cotrain 四组拓扑、Actor/Learner 优化器和 online rollout |
| [openvla_onetraj_libero_cotrain_ray_base.yaml](../../../configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml) | OFT policy、WM、classifier、replay 和数据 contract |
| [experiment/openvla_onetraj_libero_cotrain_ray.yaml](../../../configs/experiment/openvla_onetraj_libero_cotrain_ray.yaml) | public experiment 和 logger |
| [task/](../../../configs/task) | suite、checkpoint、token/hidden/action metadata |

配置覆盖顺序是：基础 config < experiment < launcher profile < launcher direct
control < `common_overrides` / `cotrain_overrides`。需要长期保留的实验设置应修改
`configs/`，一次性诊断才使用命令行 override。

任务 shorthand：

| launcher task | suite | cold-start task config |
| --- | --- | --- |
| `goal` | `libero_goal` | `openvla_onetraj_coldstart_libero` |
| `object` | `libero_object` | `openvla_onetraj_coldstart_libero_object` |
| `spatial` | `libero_spatial` | `openvla_onetraj_coldstart_libero_spatial` |
| `10` | `libero_10` | `openvla_onetraj_coldstart_libero_10` |

`token_count`、`token_dim`、`wm_obs_dim`、`chunk_size`、history、prompt、
proprio 和 image rotation 都从 task config 与 sidecar metadata 派生，不要在训练
命令中复制这些维度。

## 3. Manual Async 架构

```text
RealEnvWorker ---- observations ----+
                                    |
WMEnvWorker ------ latent obs ------+--> RolloutGroup -- trajectory --> ActorGroup
     ^                                      ^                              |
     |                                      | policy patch                 | FSDP PPO
     | WM/classifier state                  +------------------------------+
     |
LearnerGroup <-------- ReplayGroup <-------- real completed episodes
```

| Group | Worker | 职责 |
| --- | --- | --- |
| `LearnerGroup` | `LearnerWorker` | 更新 world model 和 classifier，不负责 VLA FSDP |
| `ActorGroup` | `EmbodiedFSDPActor` | VLA policy PPO、backward、optimizer 和 FSDP 通信 |
| `RolloutGroup` | `MultiStepRolloutWorker` | `eval/no_grad` policy forward，输出 action chunk、old logprob 和 forward inputs |
| `EnvGroup` | `RealEnvWorker` / `WMEnvWorker` | real/imagined stepping、episode 和 trajectory assembly |
| `ReplayGroup` | `ReplayWorker` | real replay、WM/classifier sample、WMEnv bootstrap 和可选 resume state |

### 8 卡默认 placement

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `profile=multi_gpu` 会生成：

| GPU | Env role | 其他共驻角色 |
| --- | --- | --- |
| 0 | `RealEnvWorker` | Rollout rank 0、LearnerGroup |
| 1-3 | `RealEnvWorker` | 对应 Rollout rank、ActorGroup FSDP rank |
| 4-7 | `WMEnvWorker` | 对应 Rollout rank、ActorGroup FSDP rank |

ActorGroup 使用 GPU 1-7；LearnerGroup 使用 GPU 0；RolloutGroup 在 0-7 各有一个
replica。以上角色共享 GPU，并不是每个角色独占一张卡。ReplayGroup 是 node worker，
不预留 GPU。

`osmesa` 在 CPU 渲染 real LIBERO，但 policy、WM 和 classifier 仍使用上述 GPU
placement。切换 `egl` 时必须通过 launcher 的 `render_backend=egl`，不要手工
绕过 placement 设置 `MUJOCO_EGL_DEVICE_ID`。

### 两级权重同步

1. ActorGroup 内部由 FSDP/NCCL 同步 shard、gradient 和 optimizer step。
2. 每 `manual_cotrain.sync_every=1` 个 global step，Actor 将 policy patch 发布给
   RolloutGroup。
3. 每 `manual_cotrain.learner_update_step=1` 个 global step，Learner 更新 WM 和
   classifier。
4. Learner 将 classifier 和满足更新条件的 WM state 同步给所有 WMEnvWorker。
5. policy/rollout version 记录在 `sync/` metrics；WM/classifier version 随
   trajectory sidecar 和 checkpoint manifest 保存，加载耗时/张量数记录在 `sync/`。

## 4. 数据和 Replay

collection 写入稳定的 suite 目录，不随 `RUN_ROOT` 改变：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/
  reward/
  hidden/
  collection_manifest.json
```

只有完整匹配的 reward/hidden episode pair 会计入采集进度。launcher 会隔离损坏或
不完整 shard，并按 `(task_id, episode_id)` 补采缺口。

需要区分两个 replay 生命周期：

1. DDP warmup 的 `OnlineReplay` 从上述 cold-start reward/hidden shards 完整 seed，
   用于 WM 和 classifier warmup。
2. async online 启动新的 ReplayGroup。默认由 RealEnvWorker 写入真实在线 episode；
   `manual_cotrain.wm_env_write_replay=false`，imagined episode 不回写 replay。
3. WMEnv 从 ReplayGroup 采样真实初始 `obs_embedding`、`lang_emb` 和 `proprio`；
   online 刚启动且 replay 为空时，后续 real episode 会逐步提供 bootstrap 数据。
4. warmup checkpoint 初始化 online WM/classifier 权重，但不等于自动携带 online
   ReplayGroup state。只有 `manual_cotrain.save_replay_state=true` 的 manual
   checkpoint 才能恢复 replay。

### Classifier contract

OpenVLA one-trajectory h1 主线 classifier 使用 token-WMPO 口径：

- 观测 token 来自 `*_oft_hidden_token_vla_policy_h1`，每帧形状为
  `[256,4096]`。
- `classifier.output_dim=1`。
- loss 是 BCE。
- sampling protocol 是 `wmpo`。
- balanced batch 开启。
- warmup 完成后校准 success threshold。

`classifier_warmup.ckpt` 保存校准阈值；consolidate 后该 metadata 进入
`ray_async_init.ckpt`。async Learner 的 `classifier_threshold=null` 表示优先使用
checkpoint 阈值，而不是硬编码一个新的 online threshold。

## 5. 运行前检查

在仓库根目录：

```bash
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
mkdir -p logs

bash scripts/install/60_verify.sh
```

`goal` 任务至少需要：

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1/
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1/dataset_statistics.json
${DVLA_DATA_ROOT}/datasets/libero/libero_goal/
```

wrapper 会设置 `DVLA_ROOT`、`PYTHONPATH`、`NCCL_NVLS_ENABLE=0`，并在可用时
激活 `dreamervla` conda 环境。

## 6. 先 Dry Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=osmesa dry_run=true
```

dry-run 只打印计划，不检查资产、不启动训练。应看到：

- collect: `experiment=collect_rollouts_ray`
- warmup: 8-rank `torch.distributed.run` 和
  `experiment=openvla_onetraj_libero_cotrain_noray`
- online: `experiment=openvla_onetraj_libero_cotrain_ray`
- `manual_cotrain.ngpu=8`
- `cluster.component_placement=null`，由 manual placement builder 生成拓扑

### 快速 smoke

`debug=true` 会同时缩小 collection、warmup 和 async online 预算：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=osmesa debug=true
```

### 指定 collection 总目标

launcher 会统计已有完整 episode，只补采缺口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=osmesa \
  collect_target_episodes=500 collect_num_tasks=10
```

### Sync no-Ray 基线

该入口用于同步主线基线和 Ray 问题隔离：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal render_backend=osmesa
```

## 7. 8 卡当前参数

### Collection

| 配置 | 8 卡 `multi_gpu` 值 |
| --- | ---: |
| task 数 | 10 |
| episode / task | 50 |
| 总目标 episode | 500 |
| episode horizon | 300 |
| Ray env workers | 32 |
| OFT inference workers | 4 |
| collector memory fraction | 0.9 |

### DDP warmup

warmup 与已验证的
[wm_full_dataset_train.yaml](../../../configs/experiment/wm_full_dataset_train.yaml)
对齐关键 WM 优化参数：

| 组件 | steps | local batch / GPU | 8 卡 global batch | LR |
| --- | ---: | ---: | ---: | ---: |
| world model | `20000` 配置值；10 replay epochs 重算实际 update | 16 | 128 | `3.0e-5` |
| classifier | 42 | 16 | 128 | `1.0e-4` |

其他值：

- `training.warmup_replay_epochs=10`，会按 replay sampleable windows 重算 update 数；例如 23160 是 replay epoch 推导值，不是配置里的 20000。
- `training.warmup_checkpoint_every=0`，`training.warmup_topk_k=0`：cotrain
  长跑默认只保存最终 warmup ckpt，避免 rank0 在中间 checkpoint 反复 CPU clone
  大模型造成内存峰值。
- `training.wm_profile_steps=0`：长跑默认关闭逐 update CUDA synchronize profile。
- `online_rollout.buffer_size=160000`。
- `online_rollout.sequence_length=36`。
- `world_model.chunk_rollout_chunks=4`，`chunk_rollout_loss_scale=0.2`。
- `world_model.proprio_reconstruction_loss_scale=0.0`。
- warmup 命令追加 `online_rollout.total_env_steps=0`，不会误入 sync online rollout。
- frozen OpenVLA encoder 在 replay warmup 时移到 CPU，给 WM/classifier 释放显存。

batch 16 和 WM LR `3e-5` 来自纯 WM 8 卡方案。DDP 下 batch 16 是每 rank
采样窗口数，8 卡 global update 约为 `16 * 8 = 128` 条窗口。`sequence_length=36`
和 `chunk_rollout_chunks=4` 会明显增加单次 update 的计算量，日志 profile 里约
8.3s/global update 属于这套重负载配置的量级。

### Async online

| 配置 | 当前值 |
| --- | ---: |
| real env workers | 4 |
| real env slots / worker | 8 |
| real target trajectories / global step | 32 |
| WM env workers | 4 |
| WM env slots / worker | 16 |
| WM target trajectories / global step | 256 |
| max steps / rollout epoch | 512 |
| Actor GRPO/PPO group size | 8 |
| Actor policy LR | `5.0e-7` |
| online Learner WM batch | 16 |
| online Learner WM LR | `3.0e-5` |
| online classifier batch | 2 |
| online classifier LR | `1.0e-4` |
| precision | bf16 |

online WM learner 也与 full-dataset WM recipe 对齐到 batch 16、LR `3e-5`、
`sequence_length=36`、`chunk_rollout_chunks=4` 和 `chunk_rollout_loss_scale=0.2`。
Actor 的 `5e-7` 仍用于稳定微调 VLA policy。若 async online 阶段显存不足，
优先只降低 `learner.train_cfg.batch_size`，不要改 warmup recipe。

`algorithm.group_size=8` 是 advantage 分组大小，不是 GPU 数，也不是
`dataloader.batch_size`。

### Online budget 换算

launcher 将 `online_rollout.total_env_steps` 换算为 manual global steps：

```text
global_steps =
  ceil(total_env_steps /
       (wm_rollout_target_trajectories * max_steps_per_rollout_epoch))
```

当前 `200000 / (256 * 512)` 向上取整为 2 个 manual global steps。每个 global
step 内包含 256 条大规模 imagined trajectory 的生成、Actor PPO、Learner update
和权重同步，不能把它理解为两个普通 mini-batch。

如果实验目标是增加 optimizer update 次数，应显式提高
`manual_cotrain.global_steps`，并同时设置 checkpoint/eval；不要只增大 rollout
长度来代替更新次数。

## 8. 每个 Async Global Step

当前 runner 的固定顺序：

1. 给 Actor、Rollout、RealEnv 和 WMEnv 设置 `global_step`。
2. ActorGroup 导出 policy patch，RolloutGroup 拉取新 policy。
3. RealEnv/WMEnv interaction 与 Rollout policy generation 并行执行。
4. RealEnv 完成的 episode 写入 ReplayGroup；默认只有 WMEnv trajectory 发给 Actor。
5. Actor 接收并整理 trajectory shards。
6. Actor 计算 advantages/returns。
7. ActorGroup 执行 FSDP PPO forward、backward 和 optimizer step。
8. LearnerGroup 从 real replay 各执行一次 WM 和 classifier update。
9. Learner 将 classifier/WM state 同步到 WMEnv。
10. 记录 replay、版本、训练、时耗 metrics，并按配置保存 checkpoint。

真实 LIBERO success 只看 `rollout/success_rate` 或 `eval/success_rate`。
`env/wm_env/classifier_success_rate` 是 imagined/classifier 诊断，不能作为真实成功率。

## 9. 完整 8 卡命令

下面是一条完整命令。wrapper 从 `CUDA_VISIBLE_DEVICES` 推导 `ngpu=8`，不需要
额外设置 `NGPU`。开启 in-run eval 后，launcher 会在最终 step 保存 manual
checkpoint 并执行真实 LIBERO eval。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DVLA_DATA_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal \
  render_backend=osmesa \
  run_root=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data/outputs/coldstart_warmup_cotrain/openvla_goal_$(date +%Y%m%d_%H%M%S) \
  eval.enabled=true \
  eval.interval_global_steps=10 \
  > logs/cotrain_ray_async.log 2>&1
```

如果不需要 inline eval，至少在 config 中将
`manual_cotrain.checkpoint_every` 设为正数。manual async 的基础默认值是 0，
不设置时不会周期性保存 online checkpoint。

一次性增加 online update 和保存 replay 的语法如下。online-only 键使用 `++`，
使同一 override 列表也能通过 sync warmup config 的组合：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa \
  'cotrain_overrides=["++manual_cotrain.global_steps=100","++manual_cotrain.checkpoint_every=5","++manual_cotrain.save_replay_state=true"]' \
  > logs/cotrain_ray_async_long.log 2>&1
```

长期实验应把这三个值写入
[coldstart_warmup_cotrain.yaml](../../../configs/scripts/coldstart_warmup_cotrain.yaml)
的 profile 或
[openvla_onetraj_libero_cotrain_ray.yaml](../../../configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml)，
不要依赖日志里不可追踪的临时命令。

## 10. 拆阶段运行

拆阶段时必须复用同一个 `RUN_ROOT`。

### Collection + warmup

```bash
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/openvla_goal_manual"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  cotrain_phase=warmup render_backend=osmesa \
  > logs/cotrain_warmup.log 2>&1
```

已有完整 collection 时：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  skip_collect=true cotrain_phase=warmup render_backend=osmesa \
  > logs/cotrain_warmup.log 2>&1
```

warmup 产物：

```text
${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/wm_warmup_hf/
${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/classifier_warmup_hf/
${RUN_ROOT}/cotrain/ckpt/ray_async_init.ckpt
```

`ray_async_init.ckpt` 将 WM、classifier 和校准后的 classifier threshold 合并为
manual Ray runner 可加载的单文件格式。

### Online only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  cotrain_phase=online render_backend=osmesa \
  eval.enabled=true \
  > logs/cotrain_ray_async_online.log 2>&1
```

`cotrain_phase=online` 自动跳过 collection，检查 warmup checkpoint；缺少
`ray_async_init.ckpt` 时会从两个 component warmup checkpoint 重新 consolidate。

## 11. Full-Dataset WM 预训练

完整原始数据 WM 训练继续使用独立 recipe：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPU_COUNT=8 \
DVLA_DATA_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data \
  bash scripts/experiments/world_model_training/train.sh \
  > logs/wm_train.log 2>&1
```

该 route 使用 local batch 16、global batch 128、WM LR `3e-5`、完整 timing
profile 和更长的 WM objective。cotrain `multi_gpu` warmup 已对齐其 LR/BS，
但仍保留 cotrain 自己的 1200-step cold-start warmup 和 classifier warmup。

只验证当前 8 卡 WM update 时耗优化时，使用有界一键 profile；它固定运行 64
次 update，前 32 次输出分段 timing，后 32 次用于观察没有逐步 profile 同步时的
吞吐，不会启动完整的 10-epoch 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DVLA_DATA_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data \
  bash scripts/experiments/world_model_training/profile.sh
```

如果要从独立 WM 结果继续，优先使用其
`wm_warmup_hf/` 目录作为 `init.world_model_state_ckpt`，然后仍运行 cotrain
warmup完成 classifier 校准和 `ray_async_init.ckpt` consolidation。不要把
`wm_warmup.ckpt` 直接当作 manual Ray 的三组件 init checkpoint。

## 12. 显存和吞吐调参

先跑默认 config 并记录每张卡峰值：

```bash
watch -n 2 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader'
```

建议按阶段调，不要用一个 batch 同时控制所有组件：

1. warmup OOM：将 `profiles.multi_gpu.cotrain` 中
   `dataloader.batch_size=16` 降到 14/12；classifier 可独立保留 16。
2. warmup 显存空闲：先将 WM local batch 从 16 增到 20/24，保持 LR `3e-5`
   做对照；确认 loss 和真实 eval 后再考虑 LR。
3. GPU 0 online OOM：优先减
   `learner.train_cfg.batch_size`、`classifier_batch_size` 或 real env slots。
4. GPU 4-7 online 空闲：逐步增加
   `manual_cotrain.wm_envs_per_worker`，同时保证
   `wm_rollout_target_trajectories` 能被它整除。
5. rollout 等待高但 GPU 利用低：检查 CPU osmesa、Ray channel 和 real env；
   不要先盲目增大 Actor batch。
6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 可降低碎片，但不能替代
   正确 batch 设置。

local batch 改变后，8 卡 DDP global batch 是 `local_batch * 8`。ActorGroup 是
FSDP，并不使用这条 DDP warmup batch 公式。

## 13. 时耗指标

### Warmup 每次 update

`training.wm_profile_steps`：

- `0`：关闭详细 WM update profile。
- 正整数：只 profile 前 N 次 update。
- `-1`：profile 每次 update。

详细 metrics：

```text
time/wm_warmup_sample_ms
time/wm_warmup_batch_build_ms
time/wm_warmup_data_wait_ms
time/wm_warmup_h2d_ms
time/wm_warmup_forward_ms
time/wm_warmup_backward_ms
time/wm_warmup_grad_clip_ms
time/wm_warmup_optimizer_ms
time/wm_warmup_metrics_ms
time/wm_warmup_total_ms
time/wm_warmup_device_active_fraction
```

`wm_full_dataset_train` 默认只 profile 前 8 次 update，并用 CUDA events 在每个
被 profile 的 update 末尾只同步一次。`training.wm_prefetch_workers=1` 会在当前
GPU update 期间构造下一批 replay；`sample_ms`/`batch_build_ms` 表示 CPU 工作量，
`data_wait_ms` 才是这部分工作未被 overlap 后暴露给训练循环的 stall。8 卡 cotrain
launcher 仍可把 profile 设为 0；需要临时分析更多 warmup update 时使用正整数：

```bash
warmup.wm_profile_steps=32
```

`-1` 会 profile 全部 update，只应用于短诊断，不用于正式长跑。
WM warmup 只在 `training.replay_warmup_log_every=10` 的边界把 loss 搬回 CPU；
中间 update 保持 GPU 异步执行，checkpoint 边界再按需同步。

standalone classifier 使用同一分段口径，前 8 步输出：

```text
time/classifier_update_data_wait_ms
time/classifier_update_h2d_ms
time/classifier_update_forward_ms
time/classifier_update_backward_ms
time/classifier_update_grad_clip_ms
time/classifier_update_optimizer_ms
time/classifier_update_metrics_ms
time/classifier_update_total_ms
time/classifier_update_device_active_fraction
```

另外，周期性验证和保存分别记录
`time/classifier_eval_s` 与 `time/classifier_checkpoint_s`，避免把正常的维护暂停
误判为训练 kernel 利用率问题。读数优先级如下：

- `data_wait` 高：CPU 窗口组装、replay sample、pin memory 或存储/NUMA 是瓶颈；
- `h2d` 高：检查 pinned memory、PCIe/NUMA locality；
- `backward` 相对 `forward` 异常高：重点查 DDP all-reduce 与慢 rank；
- `optimizer` 高：AdamW 状态更新受显存带宽限制；
- `device_active_fraction` 低且 eval/checkpoint 不高：仍有未归因 host stall。

`[pipeline][wm-warmup] step=... loss=...` 现在走 rank-0 event printer，8 卡 DDP
不会再由每个 rank 重复打印同一条 loss；profile/progress 也按 rank-0 口径读。

### Async global step

```text
time/manual_cotrain/set_global_step_s
time/manual_cotrain/actor_to_rollout_sync_s
time/manual_cotrain/env_interact_and_rollout_generate_s
time/manual_cotrain/actor_recv_trajectories_s
time/manual_cotrain/actor_compute_advantages_and_returns_s
time/manual_cotrain/actor_run_training_s
time/manual_cotrain/learner_update_wm_classifier_s
time/manual_cotrain/learner_to_wm_env_sync_s
time/manual_cotrain/checkpoint_and_metrics_s
```

forward 和通信细分：

```text
rollout/policy_forward_s
rollout/channel_get_s
rollout/channel_put_s
env/wm_env/wm_forward_time_s
env/wm_env/classifier_forward_time_s
env/wm_env/wm_forward_calls
env/wm_env/classifier_forward_calls
env/channel_put_obs_s
env/rollout_get_s
env/apply_step_s
actor/channel_get_batch_s
actor/load_trajectory_shards_s
sync/policy_export_s
sync/policy_push_s
sync/rollout_policy_pull_s
sync/learner_state_dicts_s
sync/wm_env_load_component_states_s
```

`actor_run_training_s` 当前覆盖 Actor forward、backward、FSDP 通信和 optimizer
总时耗；Rollout、WMEnv WM forward 和 classifier forward 有独立计时。

## 14. Metrics 判读

优先监控：

| 目标 | metrics |
| --- | --- |
| 真实策略效果 | `rollout/success_rate`、`rollout/step_success_rate`、`eval/success_rate` |
| Actor 是否更新 | `actor/ppo_updates`、`actor/loss`、`actor/policy_grad_norm`、`actor/skipped_zero_valid_update` |
| WM | `wm/loss` 及 reconstruction/cosine metrics |
| Classifier | `cls/loss`、`cls/f1`、`cls/updated` |
| Replay | `replay_buffer/size`、`replay_buffer/transitions` |
| 同步一致性 | `sync/policy_version`、`sync/rollout_policy_version`、`sync/world_model_load_s`、`sync/classifier_load_s` |
| 吞吐瓶颈 | `time/manual_cotrain/*`、`rollout/*_s`、`env/*_s`、`sync/*_s` |

完整口径见
[online cotrain metrics inventory](../../reference/metrics/online_cotrain_metrics_inventory.md)。

## 15. Checkpoint 和 Resume

启用 `manual_cotrain.checkpoint_every=N` 后：

```text
${RUN_ROOT}/cotrain/checkpoints/
  manual_cotrain_step_<N>/
    manual_cotrain.ckpt
    manual_cotrain_manifest.json
  global_step_<N>/
    manual_cotrain_manifest.json
```

`manual_cotrain.ckpt` 包含 policy、world model、classifier、threshold、global step
和 resolved config。仅当 `save_replay_state=true` 时包含 replay。

恢复必须同时完成三件事：

1. `manual_cotrain.resume_ckpt` 恢复 global step 和可选 replay。
2. `actor.init_ckpt` 从同一文件恢复 `policy`。
3. `learner.init_ckpt` 从同一文件恢复 `world_model,classifier`。

`goal` route 可使用当前用户级 resume wrapper，它会补齐上述三组配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_manual_cotrain_async.sh \
  resume=true \
  ckpt="${RUN_ROOT}/cotrain/checkpoints/manual_cotrain_step_10/manual_cotrain.ckpt" \
  gpus=0,1,2,3,4,5,6,7
```

不要只传 `manual_cotrain.resume_ckpt` 而遗漏 Actor/Learner component init。

## 16. 评估

### In-run eval

`eval.enabled=true` 会按 `eval.interval_global_steps` 分段训练。每段保存 manual
checkpoint，运行真实 LIBERO eval，并追加：

```text
${RUN_ROOT}/cotrain/eval/global_step_<N>/eval_libero_metrics.json
${RUN_ROOT}/cotrain/eval/eval_summary.json
```

### 独立 eval

```bash
CUDA_VISIBLE_DEVICES=0 \
DVLA_DATA_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/spoil/data \
  bash scripts/eval_libero_vla.sh \
  gpus=0 \
  out_dir="${RUN_ROOT}/cotrain/eval/manual" \
  +task=openvla_onetraj_coldstart_libero \
  eval.ckpt_path="${RUN_ROOT}/cotrain/checkpoints/manual_cotrain_step_10/manual_cotrain.ckpt" \
  eval.ckpt_kind=dreamer \
  eval.task_suite_name=libero_goal \
  init.vla_ckpt_path="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
```

eval 单进程运行；不要用 torchrun 启动 `EmbodiedEvalRunner`。

## 17. 日志与 W&B

```bash
tail -f logs/cotrain_ray_async.log
tensorboard --logdir "${RUN_ROOT}/cotrain/log/tensorboard" --port 6006
```

W&B 默认 offline：

```text
${RUN_ROOT}/cotrain/log/wandb/all/wandb/offline-run-*
```

共享盘同步：

```bash
bash scripts/run_wandb_relay_sync.sh --dry-run --once
bash scripts/run_wandb_relay_sync.sh --once
```

## 18. 常见故障

| 现象 | 首先检查 |
| --- | --- |
| asset check 失败 | `DVLA_DATA_ROOT`、checkpoint、`dataset_statistics.json`、suite dataset |
| collection 反复补采 | reward/hidden 是否成对，查看 `collection_manifest.json` |
| warmup OOM | local batch 16 降到 14/12，确认 frozen encoder 已移到 CPU |
| rank/NCCL timeout | 先找最早失败 rank；不要先提高 timeout 掩盖 rank divergence |
| Actor 没有 optimizer step | WMEnv trajectory 数、`actor/skipped_zero_valid_update`、advantage variance |
| Learner sample 失败 | `replay_buffer/transitions`、sequence length、classifier windows |
| WM rollout 很慢 | `wm_forward_time_s/calls`、batch avg、Rollout policy forward、GPU 利用率 |
| real rollout 很慢 | osmesa CPU 利用率、`env/apply_step_s`、`env/rollout_get_s` |
| online 没有 checkpoint | `manual_cotrain.checkpoint_every` 默认 0；启用 checkpoint 或 in-run eval |
| resume 后权重不对 | Actor、Learner 和 manual resume 是否指向同一 checkpoint |

## 19. 最终检查清单

- `dry_run=true` 的 online experiment 是
  `openvla_onetraj_libero_cotrain_ray`。
- 8 卡 warmup 显示 local batch 16、WM LR `3e-5`。
- collection manifest 达到目标且 reward/hidden pair 完整。
- `wm_warmup.ckpt`、`classifier_warmup.ckpt` 和
  `ray_async_init.ckpt` 都存在。
- online 至少保存一个 `manual_cotrain.ckpt`。
- `actor/ppo_updates > 0`，WM/classifier loss 有限。
- policy/rollout version 一致，Learner/WMEnv load metrics 持续更新。
- 最终结论使用真实 LIBERO `eval/success_rate`，不使用 imagined classifier
  success 代替。
