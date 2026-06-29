# Naming Principles

DreamerVLA 的命名目标是让长期维护者只看名称就能判断组件职责、数据方向和稳定边界。同一个概念只能存在
一个正式名称。不要为相同角色创建多个近义名称，也不要用 `V2`、`New`、`Temp`、`Manager` 等后缀掩盖
边界不清。

命名应优先表达角色，而不是表达某次实验、某个 checkpoint、某个临时实现或某位开发者的迁移路径。外部依赖
形成的名称只有在它本身就是边界时才应进入正式命名。

推荐命名具有以下特征：

- 能说明组件在 cotrain loop 中负责什么。
- 能与 Group、Worker、Replay、Env、checkpoint、metric 等公共契约对应。
- 能被配置、日志、metrics 和测试长期复用。
- 不随内部实现细节变化而变化。

不推荐命名具有以下特征：

- 表达“新旧”而不是表达职责。
- 把多个职责混进同一个名称。
- 使用某个临时路线、实验编号或历史实现作为核心概念名。
- 与已有正式名称表达同一个概念。

# Official Terms

以下名称是当前 architecture 中应优先使用的正式概念名。

| 概念 | 推荐正式名称 | 说明 |
| --- | --- | --- |
| 环境交互 group | `EnvGroup` | 管理真实环境或 WMEnv 环境交互。 |
| 真实环境 worker | `RealEnvWorker` | 负责真实 LIBERO step 和真实 rollout 轨迹。 |
| 世界模型环境 worker | `WMEnvWorker` | 负责 latent world-model environment step。 |
| 策略推理 group | `RolloutGroup` | 行为策略推理副本所在 group。 |
| 策略推理 worker | `RolloutWorker` 或 `MultiStepRolloutWorker` | 无梯度生成 action chunk 和 forward inputs。 |
| 策略训练 group | `ActorGroup` | VLA policy training、optimizer、backward、FSDP 所在 group。 |
| 策略训练 worker | `EmbodiedFSDPActor` | ActorGroup 内执行 VLA PPO 训练的 worker。 |
| 环境模型训练 group | `LearnerGroup` | 更新 world model 与 classifier/reward model。 |
| 环境模型训练 worker | `LearnerWorker` | LearnerGroup 内部 worker。 |
| 在线数据缓存 | `ReplayBuffer` 或 `OnlineReplay` | 存储、采样和 resume 数据。 |
| latent 环境 | `LatentWorldModelEnv` | world-model environment 的环境对象。 |
| 端到端运行单元 | `Runner` | 一个 train/eval job 的生命周期所有者。 |

正式名称一旦进入文档、配置、测试或日志，就应被视为公共接口的一部分。替换正式名称必须有兼容方案和验证。

# Modules

模块命名应按系统边界组织，而不是按临时任务组织。

推荐：

- `dreamervla.runners`：训练或评估 job 的入口类型。
- `dreamervla.workers.env`：环境交互 worker。
- `dreamervla.workers.rollout`：策略推理 worker。
- `dreamervla.workers.actor`：策略训练 worker。
- `dreamervla.workers.cotrain`：cotrain message、placement 和共享契约。
- `dreamervla.dataset` 或 `dreamervla.preprocess`：数据读取、转换和 sidecar 处理。
- `dreamervla.algorithms`：训练算法、actor update、reward 和 verifier 协议。

不推荐：

- `new_runner`：只表达新旧，不表达职责。
- `runner2`：无法说明边界，也会制造长期债务。
- `collector_training_manager`：把采样、训练和管理混到一起。
- `openvla_tmp_utils`：把临时实验名写入通用模块。

模块名应尽量稳定。实验特定逻辑可以通过配置、target 或 adapter 表达，不应污染长期模块名。

# Workers

Worker 命名必须表达它在 runtime 中做什么，并与所属 Group 对齐。

推荐：

- `RealEnvWorker`：真实环境 step。
- `WMEnvWorker`：world-model environment step。
- `MultiStepRolloutWorker`：多步 action chunk 推理。
- `EmbodiedFSDPActor`：FSDP VLA actor training。
- `LearnerWorker`：world model 与 classifier/reward model update。

不推荐：

- `CollectorWorker`：容易混淆 collect 阶段和长期 EnvWorker 职责。
- `TrainingEnvWorker`：环境 worker 不应训练模型。
- `InferenceWorker`：过于宽泛，无法区分 rollout inference、eval inference 或 model inference。
- `ActorWorkerV2`：用版本号代替职责说明。
- `NewLearnerWorker`：没有说明与旧 learner 的契约差异。

如果一个 Worker 同时需要支持真实环境和 WMEnv，应优先保留共同接口，再用 `RealEnvWorker` 和 `WMEnvWorker`
表达不同环境后端。不要用同一个宽泛名称隐藏两类行为的不同契约。

# Groups

Group 命名应表达并发角色和资源边界。Group 是 worker 的组织单元，不应命名为某个具体模型文件或 checkpoint。

推荐：

- `EnvGroup`：环境交互 group。
- `RolloutGroup`：行为策略推理 group。
- `ActorGroup`：VLA policy training group。
- `LearnerGroup`：world model 与 classifier/reward model training group。
- `ReplayGroup`：可选 replay service group。

不推荐：

- `PolicyGroup`：无法区分训练中的 ActorGroup 和推理中的 RolloutGroup。
- `ModelGroup`：无法说明训练哪类模型。
- `AsyncGroup`：只描述调度方式，不描述职责。
- `GPUGroup`：只描述资源，不描述角色。
- `CotrainGroup`：过宽，掩盖 group 内职责。

同一个 Group 内的 Worker 可以有多个 rank，但不能因此产生多个正式概念名。rank 是资源和并行维度，不是新的
架构角色。

# Configuration

配置命名应与架构概念一一对应。配置 key 应表达稳定契约，而不是表达某次命令行拼接或临时兼容逻辑。

推荐：

- `manual_cotrain.global_steps`
- `manual_cotrain.sync_every`
- `manual_cotrain.learner_update_step`
- `manual_cotrain.envs_per_worker`
- `actor.train_cfg.fsdp.strategy`
- `rollout.train_cfg.device`
- `learner.train_cfg.device`
- `task.openvla_oft.input_tokens.chunk_size`
- `logger.logger_backends`

不推荐：

- `new_async_steps`
- `gpu0_special_case`
- `use_old_runner`
- `infer_worker_num`
- `magic_sync`
- `tmp_checkpoint_path`

配置中不要复制 shape、dim 或 checkpoint-specific 信息。对于 OpenVLA-OFT 等路线，下游 shape 应从 task、
sidecar metadata 和 collected artifact 派生。配置 validation 应证明关系一致，而不是用隐藏默认值补齐。

# Checkpoints

checkpoint 命名必须说明组件、阶段和恢复语义。不要只用“latest”或“model”作为长期接口。

推荐：

- `wm_warmup.ckpt`：world model warmup checkpoint。
- `classifier_warmup.ckpt`：classifier/reward model warmup checkpoint。
- `global_step_<N>`：完整训练 step checkpoint 目录。
- `policy_version`、`world_model_version`、`classifier_version`：权重版本字段。
- `resolved_config.yaml`：本次运行解析后的配置事实。
- `run_manifest.json`：运行入口、路径、版本和环境信息。

不推荐：

- `final.ckpt`：无法说明是哪个组件的最终状态。
- `best_model.pt`：缺少指标来源和组件边界。
- `tmp_init.pt`：临时语义不应成为长期桥接文件名。
- `ckpt2`：无恢复语义。
- `openvla_fix.ckpt`：把一次修复写成长期名称。

checkpoint 必须能回答：保存了哪些组件、来自哪个 global step、对应哪些版本、能否完整 resume，以及缺少什么。

# Metrics

metric 命名应使用稳定 namespace，并且让读者能判断指标归属。

推荐 namespace：

- `env/`：环境 step、episode、success、reset、trajectory assembly。
- `rollout/`：action generation、old logprob、policy version、inference latency。
- `actor/`：PPO loss、advantage、ratio、entropy、optimizer step、FSDP 相关指标。
- `train/`：world model 和 classifier/reward model 的训练 loss。
- `eval/`：真实 LIBERO evaluation 指标。
- `sync/`：权重同步版本、耗时、字节量和成功状态。
- `replay_buffer/`：replay size、sample、write、resume。
- `time/`：阶段耗时和吞吐。

不推荐：

- `loss`：无法说明归属。
- `reward2`：无语义。
- `debug_metric`：临时名称泄漏进长期日志。
- `actor_env_sync_train_time`：混合多个 namespace。
- `success_new`：用新旧描述指标。

metric 名称变更会影响历史对比和监控，应视为公共接口变更。确需变更时，应保留兼容期或在文档中记录迁移。

# API

API 命名应表达动作、输入输出和同步方向。公共 API 应稳定，私有 helper 才可以随实现重构变化。

推荐：

- `set_global_step`：设置当前全局 step。
- `sync_model_from_actor`：RolloutWorker 从 ActorGroup 拉取 policy。
- `sync_model_to_rollout`：ActorGroup 发布 policy 给 RolloutGroup。
- `load_world_model_state`：WMEnvWorker 加载 world model 状态。
- `load_classifier_state`：WMEnvWorker 加载 classifier/reward model 状态。
- `collate_trajectory_shards`：合并 trajectory shard。
- `sample_batch`：从 replay 采样 batch。

不推荐：

- `do_sync`：方向和对象不明确。
- `update_all`：职责过宽。
- `process`：无法说明处理什么。
- `run2`：版本号替代语义。
- `maybe_train`：隐藏状态切换。

API 名称中若出现 `maybe`、`auto`、`magic`、`tmp`，通常说明行为不够显式。关键训练路径应避免这种命名。

# Data Contracts

数据结构命名应表达语义和 leading shape。尤其是 cotrain message、trajectory、sidecar 和 replay record，不能只用
泛化名称。

推荐：

- `ObservationMsg`
- `RolloutResultMsg`
- `TrajectoryShard`
- `TrajectoryBatch`
- `OFTRolloutBundle`
- `PixelHiddenSequenceDataset`
- `collection_manifest.json`

不推荐：

- `Data`
- `Item`
- `Blob`
- `Result`
- `Batch2`
- `HiddenStuff`

数据名称应能说明它是 message、batch、shard、record、sidecar 还是 artifact。shape 约束应写入对应数据契约
文档，而不是藏在调用方注释中。

# Deprecated Names

废弃名称应有明确处理原则。目标是保护已有用户和历史 artifact，同时避免旧名称继续扩散。

允许：

- 在兼容层读取旧 key，并在日志或 validation 中提示迁移方向。
- 在文档中标记 deprecated，说明正式名称和停止使用条件。
- 在测试中覆盖旧接口能给出明确错误或兼容行为。
- 在迁移期保留旧 artifact 的读取能力。

不允许：

- 让旧名称和新名称长期并列成为两个正式概念。
- 用 deprecated 名称创建新配置、新日志、新 metric 或新 checkpoint。
- 通过 `V2`、`legacy`、`new` 继续扩散同一概念的并行命名。
- 删除用户手写文档中的历史名称后不保留上下文。

命名迁移应遵循：先确定正式名称，再提供兼容读取，再更新主动写出的名称，最后在确认无 active 依赖后移除旧接口。

# Examples

推荐示例：

| 场景 | 推荐名称 | 原因 |
| --- | --- | --- |
| VLA 训练 group | `ActorGroup` | 表达 policy training 职责。 |
| VLA 推理 group | `RolloutGroup` | 表达行为策略采样职责。 |
| 世界模型环境 worker | `WMEnvWorker` | 表达环境后端和 worker 角色。 |
| 权重版本字段 | `policy_version` | 表达组件和版本语义。 |
| 同步指标 | `sync/actor_to_rollout_seconds` | 表达 namespace、方向和单位。 |
| replay 大小 | `replay_buffer/size` | 表达归属和含义。 |

不推荐示例：

| 场景 | 不推荐名称 | 问题 |
| --- | --- | --- |
| VLA 推理 worker | `InferenceWorker` | 过宽，无法区分 rollout、eval 或模型推理。 |
| 新 runner | `Runner2` | 用版本号代替职责。 |
| 训练入口 | `async_magic_main` | 隐藏行为，且调度方式不是职责。 |
| checkpoint | `best_model.pt` | 组件、指标和恢复语义不明确。 |
| metric | `loss` | 缺少 namespace 和模型归属。 |
| config key | `gpu0_special_case` | 把 placement 细节写成临时特殊分支。 |

命名审查时，先问三个问题：这个名称是否已有正式概念？它是否说明职责而不是实现路径？它是否能在一年后仍然成立？
