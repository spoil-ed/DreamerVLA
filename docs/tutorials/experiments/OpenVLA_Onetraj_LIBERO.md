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

# 1) 冻结 WM/CLS，不做真实环境 eval
WORLD_MODEL_CKPT=/path/to/world_model/run-or-checkpoint \
CLASSIFIER_CKPT=/path/to/classifier/run-or-checkpoint \
  bash scripts/e2e_frozen_model_cotrain.sh

# 2) 冻结 WM/CLS；step 0 评测基础 VLA，之后每 10 global_step 评测 PPO-VLA
WORLD_MODEL_CKPT=/path/to/world_model/run-or-checkpoint \
CLASSIFIER_CKPT=/path/to/classifier/run-or-checkpoint \
  bash scripts/e2e_frozen_model_cotrain_eval.sh

# 3) WM/CLS 继续训练；使用相同 PPO 与相同周期 VLA eval
WORLD_MODEL_CKPT=/path/to/world_model/run-or-checkpoint \
CLASSIFIER_CKPT=/path/to/classifier/run-or-checkpoint \
  bash scripts/e2e_wmcls_cotrain_eval.sh

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

两个带 eval 的 recipe 固定采用 `libero_goal` 的 task `0..9`，每个 task
依次评测 init state `0..9`，因此一次 eval 恰好是 100 个真实 episode。
`global_step=0` 直接评测
`${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1`；
后续 checkpoint 加载 ActorGroup 保存的完整 OpenVLA-OFT policy，包括已经 SFT 的
vision backbone/projector 和 PPO 更新的原生 LM/OFT action decoder。不能把 learned
actor 接回固定 base encoder，否则 latent space 错配，评估结果没有因果意义。评测不写
replay、不进入 PPO batch，也不参与 WM/CLS 拟合或 threshold 校准。每个训练 global
step 的正式超时为 5400 秒；旧的 600 秒会在
`1024 x 512` rollout 仍正常前进时误报超时。
训练进度里的 `wm_cls_chunk_positive_rate` 与
`wm_cls_trajectory_positive_rate` 只是 imagined rollout 上 classifier 超阈值的
比例，不是真实环境 SR；真实 SR 只看周期 eval 的 `eval_success_rate` 和逐 task 指标。

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
| [experiment/dreamervla_frozen_models_rl_ray.yaml](../../../configs/experiment/dreamervla_frozen_models_rl_ray.yaml) | 冻结 WM/CLS 的 8 卡 policy-only 测试 experiment |
| [experiment/dreamervla_frozen_models_rl_ray_eval.yaml](../../../configs/experiment/dreamervla_frozen_models_rl_ray_eval.yaml) | 冻结 WM/CLS、step 0/每 10 step 真实 VLA eval |
| [experiment/dreamervla_wmcls_cotrain_ray_eval.yaml](../../../configs/experiment/dreamervla_wmcls_cotrain_ray_eval.yaml) | WM/CLS 可训练、step 0/每 10 step 完整 VLA + causal diagnostics eval |
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
raw image/text/proprio
    -> OpenVLA vision backbone + projector
    -> visual tokens [token_count, token_dim]
    -> native text/proprio concat + OpenVLA LM/OFT decoder
    -> action-token distribution -> actions

RealEnvWorker -- 32 completed raw trajectories --> ActorGroup encoder SFT
                                                      |
                                                      +--> re-encode all 32
                                                               |
                                                               v
LearnerGroup <------ replace-only ReplayGroup <------ current-step latents
     |
     +-- latest WM/classifier/threshold --> WMEnvWorker
                                              |
                                              v
                                      imagined trajectories
                                              |
                                              v
                                     ActorGroup actor PPO

ActorGroup -- full-policy patch --> RolloutGroup -- no-grad raw/latent inference
```

主线 policy 始终是原始 OpenVLA-OFT。当前 checkpoint 的 projected visual sidecar
是 `[256,4096]`，但这两个值来自 checkpoint 与 task metadata，不是写死的 policy
结构。raw path 由当前 encoder 产生这些 token；latent path 接受 WM 预测的同空间
token，两条路径随后共用 checkpoint 原生的 prompt、attention mask、text/proprio
拼接、action-token 位置和解码逻辑。主线不构造随机 Transformer bridge 或 56 个
learned action queries。

| Group | Worker | 职责 |
| --- | --- | --- |
| `LearnerGroup` | `LearnerWorker` | 更新 world model 和 classifier，不负责 VLA FSDP |
| `ActorGroup` | `EmbodiedFSDPActor` | 完整 VLA；成功真实轨迹的 encoder SFT、imagined-only actor PPO、两个 optimizer 和 FSDP 通信 |
| `RolloutGroup` | `MultiStepRolloutWorker` | `eval/no_grad` 完整 policy 副本；real 用 raw path、WM 用 latent path |
| `EnvGroup` | `RealEnvWorker` / `WMEnvWorker` | real/imagined stepping、episode 和 trajectory assembly |
| `ReplayGroup` | `ReplayWorker` | 每步 replace-only real replay、WM/classifier sample 和 WMEnv bootstrap |

冻结 WM/CLS 的 pre-mainline Ray 实验不实现另一套 RL：它同样调用
`ManualCotrainRayRunner`，复用相同的 `LatentWorldModelEnv`、
`MultiStepRolloutWorker`、`LatentToOpenVLAHiddenStateActor` 和 Actor
PPO/FSDP 路径。区别是 Hydra 将
`manual_cotrain.learner_updates_enabled=false` 且 `learner=null`，WM/CLS
从显式 checkpoint 加载后保持冻结。这里的 `LatentToOpenVLAHiddenStateActor` 是
隔离的 legacy feasibility policy，不是非冻结主线 policy。非冻结 staged 主线由
LearnerGroup 更新 WM/CLS，并由完整 `OpenVLAOFTPolicy` 同时拥有 encoder 与原生
action decoder；两条路线只共享 WM/classifier 和 worker/channel 基础设施，不应混用
policy checkpoint。

### 8 卡默认 placement

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `profile=multi_gpu` 会生成：

| GPU | Env role | 其他共驻角色 |
| --- | --- | --- |
| 0 | `RealEnvWorker` | Rollout rank 0、Actor rank 0、LearnerGroup |
| 1-3 | `RealEnvWorker` | 对应 Rollout rank、ActorGroup FSDP rank |
| 4-7 | `WMEnvWorker` | 对应 Rollout rank、ActorGroup FSDP rank |

ActorGroup 和 RolloutGroup 都在 GPU 0-7 各有一个 rank；LearnerGroup 与 GPU 0
上的 Actor/RealEnv/Rollout 共卡。这样非冻结主线与冻结实验保持相同的 8-rank Actor
PPO/FSDP 结构，唯一差异是前者存在 LearnerGroup 更新 WM/CLS。ReplayGroup 是 node
worker，不预留 GPU。

`osmesa` 在 CPU 渲染 real LIBERO，但 policy、WM 和 classifier 仍使用上述 GPU
placement。切换 `egl` 时必须通过 launcher 的 `render_backend=egl`，不要手工
绕过 placement 设置 `MUJOCO_EGL_DEVICE_ID`。

### 两级权重同步

1. ActorGroup 内部由 FSDP/NCCL 同步 shard、gradient 和 optimizer step。
2. staged step 开始时，Actor 将 step-entry `pi_old` 发布给 RolloutGroup，供真实
   collection 使用。
3. encoder SFT 通过 KL gate 后，再发布 post-SFT policy，供 imagined action inference
   使用。
4. 当前步 latent 上完成 WM/CLS 多步拟合和 threshold 重校准后，Learner 将三者同步给
   所有 WMEnvWorker；同步 barrier 之后才允许 imagined rollout。
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

需要区分两个数据生命周期：

1. DDP warmup 的 `OnlineReplay` 从上述 cold-start reward/hidden shards 完整 seed，
   用于 WM 和 classifier warmup。
2. async staged online 每个 global step 收集恰好 32 条完整真实轨迹；成功轨迹先用于
   encoder SFT，随后全部 32 条都由新 encoder 重编码。
3. ReplayGroup 对当步重编码 batch 执行 `replace`，而不是 append。WM/CLS 参数与
   optimizer state 跨步延续，但训练样本不跨 global step 保留；如果当步没有可用
   数据则显式失败/跳过，不会借用旧 replay。
4. `manual_cotrain.wm_env_write_replay=false`，imagined episode 不回写 replay，真实
   trajectory 也不进入 Actor PPO。WMEnv bootstrap 只来自当前步 re-encoded history。
5. 默认 `save_replay_state=false`，full checkpoint 不保存 step-local 样本；resume 后
   重新收集当前步真实轨迹。warmup checkpoint 只负责初始化 online WM/CLS 权重。

### Classifier contract

OpenVLA one-trajectory h1 主线 classifier 使用 token-WMPO 口径：

- 观测 token 来自 `*_oft_hidden_token_vla_policy_h1`，每帧形状为
  `[256,4096]`。
- `classifier.output_dim=1`。
- loss 是 BCE。
- sampling protocol 是 `wmpo`。
- balanced batch 开启。
- warmup 完成后校准 success threshold。

`classifier_warmup.ckpt` 保存初始校准阈值；consolidate 后该 metadata 进入
`ray_async_init.ckpt`。每个 staged online step 结束 WM/CLS 拟合后，会用当步数据重新
校准 threshold；如果 calibration split 只有单一类别，则保留上一阈值并报告 skip。
固定 100 条真实 eval 只读取该阈值，不参与校准。

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
- 独立的 `wm_full_dataset_train` / `wm_official_upper_bound` 离线 WM recipe
  还设置 `training.world_model_ddp={find_unused_parameters: false,
  broadcast_buffers: false, static_graph: true, gradient_as_bucket_view: true}`。
  该 recipe 每次只执行同一个 chunk-loss 图：保留但 loss scale 为 0 的 reward
  head 构成固定 unused 集合。static graph 保留这个语义，同时去掉每步 autograd
  图搜索；bucket view 去掉 gradient 到 NCCL bucket 的额外复制。同步 online route
  会穿插 WM 的多种 forward mode，因此不继承这个离线专用开关。
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
| WM target trajectories / global step | 1024 |
| max steps / rollout epoch | 512 |
| Actor GRPO/PPO group size | 8 |
| Actor global batch（flattened chunk samples） | 16384 |
| Actor micro batch / rank | 32 |
| Actor optimizer steps / global step | 4 |
| Actor policy LR | `5.0e-7` |
| encoder SFT epochs / batch | `2` / `4` |
| encoder LR | `1.0e-7` |
| cumulative policy KL budget | `0.10` |
| online Learner WM batch | 16 |
| online Learner WM LR | `3.0e-5` |
| online classifier batch | 2 |
| online classifier LR | `1.0e-4` |
| max WM/CLS update iterations / early-stop patience | `128` / `16` |
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

当前 `200000 / (1024 * 512)` 向上取整为 1 个 manual global step。每个 global
step 先完整生成 1024 条、每条 512 physical steps 的 imagined trajectory；OFT
`chunk_size=8` 后得到 `1024 * 64 = 65536` 个 PPO sample。Actor 按 global batch
16384 切成 4 个 optimizer batch，每个 rank 再按 micro batch 32 累积梯度，因此一个
manual global step 明确包含 4 次 PPO optimizer step。注意 imagined rollout 与 PPO
之前已经完成真实 collection、encoder SFT、全量重编码、WM/CLS step-local 更新、
threshold 校准和 Learner-to-WMEnv 同步。

如果实验目标是增加 optimizer update 次数，应显式提高
`manual_cotrain.global_steps`，并同时设置 checkpoint/eval；不要只增大 rollout
长度来代替更新次数。

## 8. 每个 Async Global Step

当前 runner 的固定顺序：

1. 给 Actor、Rollout、RealEnv 和 WMEnv 设置 `global_step`。
2. ActorGroup 发布 step-entry `pi_old`，RolloutGroup 拉取完整 policy。
3. 先在 global-step 边界 reset 全部真实 slot，丢弃任何旧 policy 留下的半条轨迹；
   `pi_old` 再在真实 LIBERO 中收集并 drain 恰好 32 条完整 trajectory。每个
   slot/rollout epoch 在第一条终止轨迹后停止，因此提前成功也不会额外采样；batch
   保留 raw image、task text/proprio、采样 action-token ID、实际动作和成功标签。
4. 只用成功 trajectory 的 policy decision 做 encoder-only SFT。actor decoder 冻结；
   用完整 action-token distribution 测量 SFT 前后 KL。若超过总预算，policy 与 encoder
   optimizer 在重编码前回滚。
5. 用已接受 encoder 重编码全部 32 条成功/失败 trajectory，并以同一步
   `encoder_version` 替换 ReplayGroup 内容。
6. LearnerGroup 在该 step-local replay 上最多进行 128 次 WM+CLS 更新，patience 16
   early stop；随后重新校准 classifier threshold。WM 的 one-step/token MSE、cosine，
   以及 4 个 action chunk 的 closed-loop rollout MSE/cosine 都进入训练日志。
7. Learner 将最新 WM、classifier 和 threshold 同步到 WMEnv；WMEnv 从当前步 history
   初始化。Actor 再发布 post-SFT policy。
8. 最新 WM 完全 closed-loop 生成 1024 条 imagined trajectory；只有这些 trajectory
   发送给 ActorGroup。
9. encoder 冻结。Actor 计算 advantage/return，并按 global batch 16384、每 rank micro
   batch 32 执行 imagined-only FSDP PPO，共 4 次 optimizer step。PPO 只能使用
   `max_policy_kl - encoder_KL`；越界时 policy 与 actor optimizer 回滚到 post-SFT。
10. 在完成的 transaction boundary 保存完整 checkpoint；launcher 默认在 step 0 和
    之后每 10 step 运行固定 100 条真实轨迹的只读 causal eval，再从该 checkpoint
    resume 下一段。

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

一次性增加 online update 和 checkpoint cadence 的语法如下。online-only 键使用 `++`，
使同一 override 列表也能通过 sync warmup config 的组合：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa \
  'cotrain_overrides=["++manual_cotrain.global_steps=100","++manual_cotrain.checkpoint_every=1"]' \
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
actor/ppo_optimizer_steps
actor/ppo_forward_backward_steps
actor/global_ppo_samples
actor/global_batch_size
actor/micro_batch_size
actor/policy_loss
actor/approx_kl
actor/clip_fraction
actor/lr
sync/policy_export_s
sync/policy_push_s
sync/rollout_policy_pull_s
sync/learner_state_dicts_s
sync/learner_state_share_s
sync/wm_env_load_component_states_s
```

`actor_run_training_s` 当前覆盖 Actor forward、backward、FSDP 通信和 optimizer
总时耗；Rollout、WMEnv WM forward 和 classifier forward 有独立计时。
trainable WM/CLS 同步时，`learner_state_dicts_s` 是 learner 生成 CPU snapshot 的
时间，`learner_state_share_s` 是把同一 snapshot 放入 Ray object store 一次的时间，
`wm_env_load_component_states_s` 是所有 WMEnv worker 完成加载的端到端等待时间。
三者分开后，不再把 8 worker 广播序列化误判成模型更新时间。

## 14. Metrics 判读

优先监控：

| 目标 | metrics |
| --- | --- |
| 真实策略效果 | `rollout/success_rate`、`rollout/step_success_rate`、`eval/success_rate` |
| Actor 是否更新 | `actor/ppo_optimizer_steps`、`actor/ppo_forward_backward_steps`、`actor/policy_loss`、`actor/grad_norm`、`actor/lr` |
| PPO 稳定性 | `actor/ratio`、`actor/ratio_abs`、`actor/approx_kl`、`actor/clip_fraction`、`actor/entropy_mean` |
| PPO 数据契约 | `actor/global_rollout_trajectories`、`actor/global_ppo_samples`、`actor/global_loss_mask_sum`、`actor/global_batch_size`、`actor/micro_batch_size` |
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

`manual_cotrain.ckpt` 包含完整 VLA policy、actor optimizer、encoder optimizer、
world model、classifier、两个 Learner optimizer、threshold、global step、metrics 和
resolved config。默认 `save_replay_state=false`，不保存当步 32 条训练样本；只保存
replay sampling/version state。staged resume 会从下一步重新收集真实轨迹。

恢复必须同时完成三件事：

1. `manual_cotrain.resume_ckpt` 恢复 global step 与完整 payload；launcher 动态注入时使用
   `++manual_cotrain.resume_ckpt=...`，兼容未预声明该键的 Hydra struct config。
2. Actor 从同一文件恢复 `policy`、policy optimizer 和 encoder optimizer。
3. Learner 从同一文件恢复 `world_model`、`classifier`、两个 optimizer 和 threshold。

`goal` route 可使用当前用户级 resume wrapper，它会补齐上述三组配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_manual_cotrain_async.sh \
  resume=true \
  ckpt="${RUN_ROOT}/cotrain/checkpoints/manual_cotrain_step_10/manual_cotrain.ckpt" \
  gpus=0,1,2,3,4,5,6,7
```

标准 segmented launcher 会自动完成上述映射；手工直启 runner 时不要只恢复某一个
component。

## 16. 评估

### In-run eval

`eval.enabled=true` 会按 `eval.interval_global_steps` 分段训练。每段保存 manual
checkpoint，运行真实 LIBERO eval，并从这个完整 checkpoint 恢复下一段。主线默认
interval 为 10；协议固定为 task `0..9`、每 task 10 条，共 100 条 trajectory。
评估 observer 逐条流式编码，不写 replay、不执行 backward，也不改变 threshold。
除真实 VLA SR 外，它报告 real-latent classifier trajectory F1/precision/recall/
accuracy/PR-AUC/ROC-AUC，以及从前 `H=3` 个真实 latent 初始化、之后对每个完整
`K=8` action chunk 递归预测的 WM horizon MSE/cosine 和 WM->classifier 同组指标。
结果写入：

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
  eval.ckpt_kind=vla_policy \
  eval.cotrain_diagnostics=true \
  eval.cotrain_expected_trajectories=100 \
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
- `actor/ppo_optimizer_steps > 0`，WM/classifier loss 有限。
- policy/rollout version 一致，Learner/WMEnv load metrics 持续更新。
- 最终结论使用真实 LIBERO `eval/success_rate`，不使用 imagined classifier
  success 代替。
