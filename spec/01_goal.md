# Project Goals

DreamerVLA 的长期目标是形成一个可持续维护、可验证、可恢复的单机 VLA 与世界模型协同训练系统。
系统主线不是堆叠更多训练入口，而是把 collect、warmup、cotrain、eval 组织成边界清晰的训练闭环，
让每一类模型、数据、权重和指标都能说明自己的来源、用途、版本和落盘位置。

`spec/` 是 architecture source of truth。代码应逐步向这里描述的目标架构收敛；当代码事实与目标
架构不一致时，应先记录为 `Current vs Target` 或 `Open Questions`，再决定修改文档还是修改实现。
不要把临时实现细节反向写成长期架构原则。

DreamerVLA 的主线目标包括：

- 用真实 LIBERO rollout 提供初始数据和最终评估锚点。
- 用 world model 和 classifier/reward model 扩展在线训练中的环境侧信号。
- 用 ActorGroup 训练 VLA policy，用 RolloutGroup 提供无梯度行为策略推理。
- 用 LearnerGroup 独立更新 world model 与 classifier/reward model。
- 用稳定 replay artifact、checkpoint、metrics 和日志支撑 resume、debug 和长期运行。
- 让 0 到多 GPU 的启动形态遵循同一套职责边界，而不是为不同机器写不同架构。

本项目追求的是长期可演进的 RL 系统，而不是一次性跑通的脚本集合。任何新增能力都应优先融入现有
Runner、Group、Worker、Hydra config、数据契约和验证矩阵。

# Architecture Principles

## Single Responsibility

每个核心组件只承担一个主要职责。职责划分应能用一句话说明，并且能被独立测试。

推荐边界：

- EnvWorker 负责环境交互、slot 状态管理和 trajectory assembly。
- RolloutWorker 负责当前行为策略的无梯度推理。
- ActorGroup 负责 VLA policy 的训练、反传、优化器和 FSDP 通信。
- LearnerGroup 负责 world model 与 classifier/reward model 的更新。
- ReplayBuffer 负责数据存储、采样、resume 和数据来源记录。
- Evaluator 负责真实环境评估和指标统计。

避免让单个组件同时承担 collect、learning、checkpoint、logging、evaluation、synchronization 等多个
核心职责。一个组件如果既在采样又在训练，又负责保存模型和解释指标，通常说明边界已经失效。

## Explicit Over Implicit

系统行为应尽可能显式。关键状态变化必须能从配置、日志、metrics 或 checkpoint 中追踪到。

需要显式表达的状态包括：

- 当前 global step、episode id、task id 和 slot id。
- policy、world model、classifier/reward model 的版本。
- 权重同步的来源、目标、触发时机和成功结果。
- replay 数据的来源、shape、sidecar key 和写入位置。
- checkpoint 包含的组件、版本和恢复语义。
- placement 中每个 group 与 worker 使用的资源。

避免隐式同步、隐式状态切换、隐式资源占用和隐式副作用。尤其不要让某个训练函数在没有日志或版本记录
的情况下修改 rollout policy、world model env 或 replay 状态。

## Stable Public Interface

公共接口应长期稳定。Runner、Worker、Group、Replay、dataset、checkpoint 和 metrics 的外部契约一旦被
配置、脚本、测试或文档依赖，就不应被随意删除或改名。

需要调整接口时，优先采用：

- 新增字段并保留旧字段的兼容读取。
- 新增兼容 adapter，而不是直接改动上游调用方。
- 在文档中标明 deprecated 状态、迁移方向和停止使用条件。
- 用测试覆盖旧接口仍能被加载或给出明确错误。

稳定接口不等于冻结内部实现。内部可以重构，但外部契约必须有迁移路径。

## Minimal Abstraction

抽象层只在降低真实复杂度时才值得引入。新增抽象必须能带来至少一种明确收益：

- 消除重复且易错的跨模块逻辑。
- 固定长期公共契约。
- 隔离训练后端、模型实现或数据来源差异。
- 让测试可以在不启动完整系统的情况下验证核心行为。

不要为了“未来可能需要”引入空泛 manager、helper、factory 或 versioned 类。能用已有 Hydra target、
registry、protocol 或 dataclass 表达的，不应再新增并行抽象。

## Config As Contract

Hydra config 是运行时契约来源。模型尺寸、sidecar key、batch shape、checkpoint 路径、logger 后端、
precision、placement 和训练开关都应从配置进入系统，再由 validation 证明关系成立。

validation 的职责是拒绝不一致的配置，不是偷偷选择训练行为。隐藏默认值会让实验难以复现，也会让
resume 和长期运行变得不可解释。

# System Boundaries

以下边界是长期不可破坏的系统边界。

ActorGroup 与 RolloutGroup 必须分离。二者可以使用同一类 VLA 架构，但角色不同：ActorGroup 是学习中的
policy，承担 optimizer、backward 和 FSDP；RolloutGroup 是行为策略推理副本，处于 eval/no-grad 状态，
只用于根据观测生成 action chunk 和可复算 logprob 的 forward inputs。

LearnerGroup 不训练 VLA。它只负责 world model 与 classifier/reward model。若未来需要额外环境侧模型，
也应沿 LearnerGroup 的环境模型职责扩展，而不是把 policy update 塞进 LearnerGroup。

EnvGroup 不训练模型。RealEnvWorker 负责真实环境 step，WMEnvWorker 负责 latent world-model environment
step。它们可以加载用于推理的 world model/classifier 副本，但这些副本只能通过显式同步更新。

Replay 不是 ActorGroup PPO 的隐藏替代通道。Replay 用于存储、resume、warmup、world-model/classifier
训练和 WMEnv bootstrap；ActorGroup 的 PPO 数据应来自 rollout 中显式组装的 trajectory channel。

Warmup checkpoint bridge 与 manual cotrain checkpoint 都必须有清晰组件边界。world model、classifier、
VLA policy、global step、版本号和 resolved config 不能混成不可解释的单一文件。

`99_manual_notes.md` 是用户第一性指导。除非用户明确要求，不得移动、压缩、重写或删除其中的手写内容。

# Runtime Invariants

以下不变量默认始终成立。实现若暂时无法满足，必须在相关文档中明确说明原因、影响和恢复路径。

- collect 可以与 learner update 在架构上并行，不能把采样和学习写死成互斥流程。
- replay artifact 必须支持 resume，并能解释每条核心数据的来源与版本。
- actor、rollout、world model、classifier/reward model 权重必须具备版本概念。
- 权重同步必须有显式触发点、方向、版本和结果记录。
- checkpoint 必须支持完整恢复训练，而不是只保存某个模型的裸参数。
- metrics namespace 必须稳定，不能因内部重构频繁改名。
- 所有关键数据路径必须可追踪，包括 collect 输出、hidden sidecar、reward shard、replay、checkpoint 和视频。
- trajectory 中用于训练的 action 与用于环境 step 的 action 必须表示同一个 chunk。
- `forward_inputs` 必须足够让 ActorGroup 对同一 action chunk 重新计算 logprob。
- 真实 LIBERO eval 是最终质量锚点，WMEnv reward 不能替代最终评估。

这些不变量服务于长期调试和科学复现。只要无法说明某个 tensor、metric 或 checkpoint 从哪里来、要到哪里去，
就不应把它视为稳定架构的一部分。

# Maintenance Principles

长期维护优先级是：先保持系统可解释，再追求局部性能；先稳定公共契约，再重构内部实现；先补验证，再扩大
训练规模。

维护时应遵循：

- 新路线必须通过 Runner、Hydra config 和测试进入系统。
- 新模型、数据集、reward、verifier 或 actor update 应通过现有 registry、target 或 protocol 表达。
- 跨模块行为应有窄接口，不应靠共享全局状态或临时环境变量传递。
- 运行产物必须落在当前 run root 或配置声明的数据根下。
- 脚本只做薄封装，业务逻辑应进入 Python/Hydra 层。
- 历史计划和旧实验只能作为参考，不能覆盖当前主文档和用户第一性指导。

当文档之间出现冲突时，优先顺序是：`99_manual_notes.md` 的第一性判断、当前代码与测试事实、当前主文档的
明确 architecture、历史计划和旧方案。

# Validation Philosophy

所有重要修改都应有验证证据。验证不是为了追求形式完整，而是为了证明系统边界没有被破坏。

推荐验证层级：

1. import check：证明模块、Hydra target 和依赖路径可加载。
2. unit test：证明单个契约、shape、命名、配置或同步行为正确。
3. tiny smoke：用最小配置启动目标 Runner，并完成短流程。
4. GPU smoke：验证 placement、device、precision、FSDP 或 Ray 资源路径。
5. LIBERO end-to-end：验证真实环境 collect、cotrain 或 eval 路径。
6. long-run stability：验证 checkpoint、resume、metrics、memory 和吞吐在长时间运行下稳定。

新增功能、大规模重构、权重同步、checkpoint 语义、trajectory shape 和 actor/learner 边界修改后，不应只依赖
静态阅读判断正确性。没有可重复验证命令的修改，只能算未完成风险。

# Refactor Rules

重构的目标是让代码向 specification 收敛，同时保持行为可解释、可回退、可验证。

重构时应遵循：

- 优先保持行为一致，再改变结构。
- 优先增量修改，不做无必要的大规模重写。
- 优先收敛命名和边界，不引入并行概念。
- 优先 deprecated，再 remove。
- 每次重要修改后运行对应验证。
- 不删除用户手写内容。
- 不把目标、现状和历史计划混写在同一段结论里。
- 不把临时兼容逻辑包装成长期架构原则。
- 不让新抽象绕过 Hydra、Runner、Worker、Replay、checkpoint 或 metrics 的现有契约。

如果发现现有实现与 specification 不一致，应先判断它是代码欠债、文档过时，还是未确认目标。只有确认来源后，
才更新实现或文档。
