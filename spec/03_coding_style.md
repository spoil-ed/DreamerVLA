# Coding Style

DreamerVLA 的编码风格服务于长期维护大型 RL 系统。代码应让训练流程、数据流、同步边界和恢复语义清晰可查，
而不是把复杂性藏进宽泛 helper、隐式状态或临时脚本。

所有实现应遵循五个基本原则：

- 单一职责：一个类、函数或模块只承担一个主要职责。
- 显式优于隐式：关键状态变化、数据路径和同步行为必须可追踪。
- 稳定公共接口：对外契约要有兼容和迁移路径。
- 最小抽象：只在减少真实复杂度时新增抽象。
- 向后兼容：优先 deprecated 和 adapter，而不是直接删除。

本文件描述长期风格约束，不替代具体 API contract、shape contract 或验证矩阵。具体数据结构、checkpoint、
metrics 和 placement 约束应写入对应 spec 文档。

# Class Design

类应围绕单一角色设计。类名说明职责，构造参数说明依赖，public method 说明可被外部依赖的行为。

推荐：

- Runner 拥有一个 train/eval job 的生命周期：setup、execute、teardown、checkpoint、metrics。
- Worker 拥有一个 runtime 角色：环境交互、策略推理、策略训练或环境模型训练。
- Env 对象只表达环境 step/reset 语义，不负责训练、checkpoint 或全局调度。
- Replay 对象只表达数据写入、采样、resume 和 artifact 管理。
- Helper 只封装局部、可测试、无隐藏状态的逻辑。

不推荐：

- 一个类同时负责构建模型、采样、训练、保存、评估和同步。
- 一个类通过多个 mode 字符串在完全不同职责之间切换。
- 一个 manager 持有过多 group，并直接修改它们的内部状态。
- 一个 utility 模块逐渐积累训练业务逻辑，变成隐藏入口。

当类开始需要解释“在这个场景下它不是做这个，而是做另一个职责”时，通常应重新划分边界。

继承应谨慎使用。只有当子类真正共享接口并能替换父类时才使用继承；否则优先组合、protocol、dataclass 或
Hydra target。不要用继承表达临时实验差异。

# Interface Design

公共接口应小而稳定。接口的输入、输出、副作用和错误条件必须能被测试和文档说明。

接口设计要求：

- 方法名表达动作和方向，例如从哪里同步到哪里、从哪里采样、加载哪个组件。
- 输入参数使用明确类型，不用宽泛对象承载多种含义。
- 输出结构稳定，新增字段应保持兼容。
- 副作用必须可预期，尤其是权重更新、replay 写入、checkpoint 写入和日志记录。
- 外部可调用接口不应依赖调用方知道内部初始化顺序之外的隐式状态。

推荐做法：

- 用 `setup` 或 `init` 明确资源构建边界。
- 用 `execute` 或 `run` 表示一个可观测阶段。
- 用 `load_*_state` 表示只加载状态，不触发训练。
- 用 `sync_*` 表示跨组件版本同步，并返回版本或 metrics。
- 用 `validate_*` 表示只检查契约，不改变训练行为。

不推荐做法：

- `process`、`handle`、`update_all` 等宽泛方法承载多种流程。
- 方法根据全局变量或环境变量静默改变训练路径。
- validation 中偷偷填充训练默认值。
- 公共 API 返回未说明 shape 的裸 tuple。

接口稳定不意味着永远不能改。需要变更时，应先新增兼容字段或 adapter，再逐步迁移调用方，并用测试覆盖迁移期行为。

# Structured Data

跨模块数据应优先使用结构化表示。cotrain message、trajectory、checkpoint manifest、collection manifest、
sidecar metadata 和 metrics payload 都应有清晰字段语义。

推荐：

- 用 dataclass 表达轻量、稳定、进程内传递的数据。
- 用 typed dict、schema 或 manifest 表达落盘 JSON/YAML artifact。
- 用明确字段保存 `task_id`、`episode_id`、`slot_id`、`global_step`、版本和 shape 信息。
- 对 tensor leading dimensions 使用稳定约定，并在数据契约文档中说明。
- 对可选字段明确说明何时为空、谁负责填充、下游如何处理。

不推荐：

- 用 `dict[str, Any]` 在多个模块之间传递未说明结构的核心数据。
- 让调用方靠字段是否存在猜测当前训练分支。
- 把 shape 约束只写在某个局部注释里。
- 在 artifact 中保存无法解释来源或版本的 tensor。

结构化数据不是为了增加样板，而是为了让长期维护者能追踪数据来源、流向和恢复语义。任何跨 worker、跨进程、
跨 checkpoint 的数据都应优先结构化。

# Logging

日志应解释关键状态变化，而不是倾倒内部细节。训练循环中的日志要稳定、低噪声，并能与 metrics 和 checkpoint
互相对照。

应记录：

- Runner 启动入口、resolved config 路径和 run root。
- Group/Worker 拓扑、placement 和设备信息。
- 权重同步方向、版本、耗时和结果。
- replay 读写路径、样本数量和 resume 状态。
- checkpoint 保存路径、组件和 global step。
- 关键异常的上下文，包括配置 key、组件名和 artifact 路径。

不应记录：

- 每个 step 的大量裸 tensor。
- 无 namespace 的临时 debug 行。
- 会频繁变化且无法用于诊断的内部对象 repr。
- 训练循环中的大量 bare print。

训练代码应优先使用 runner logging、JSON logger、TensorBoard 或 W&B 后端。少量 rank-0 progress 行可以保留，
但不能成为唯一可追踪信息来源。

# Metrics

metrics 是长期分析接口，应比日志更稳定。新增 metric 前应确认 namespace、单位、聚合方式和 owner。

要求：

- 使用稳定 namespace：`env/`、`rollout/`、`actor/`、`train/`、`eval/`、`sync/`、`replay_buffer/`、`time/`。
- 名称表达单位或语义，例如 seconds、count、rate、loss、version。
- 同一指标不要在多个 namespace 下重复出现。
- 内部重构不应随意改 metric 名称。
- 临时诊断指标应有明确前缀或只写 diagnostics artifact，不应污染长期训练曲线。

metrics 应通过 Runner 或统一 logger 路由。不要让 Worker 自行决定外部日志后端。

# Exceptions

异常处理应快速暴露契约错误，并携带足够上下文。训练系统中最危险的错误不是失败，而是带着错误配置继续运行。

推荐：

- 配置缺失、shape 不匹配、sidecar key 缺失、checkpoint 组件缺失时尽早报错。
- 错误信息包含组件名、配置 key、期望值、实际值和相关路径。
- 对可恢复外部状态使用明确 fallback，并记录 fallback 原因。
- 对未实现的可选分支使用明确错误，而不是静默跳过。

不推荐：

- 捕获所有异常后只打印字符串继续训练。
- 用默认空 tensor、空 dict 或零值掩盖数据缺失。
- 在分布式训练中只让非 rank0 报错，导致 rank0 卡住。
- 把配置问题推迟到深层训练循环才暴露。

异常消息也是公共体验的一部分。它应帮助维护者定位问题，而不是只说明“失败了”。

# Synchronization

同步必须显式、分层、可记录。DreamerVLA 至少存在三类不同同步，不能混为一谈。

ActorGroup 内部同步由 FSDP/NCCL 管理。它属于 policy training 内部机制，不应通过手写权重复制替代。

ActorGroup 到 RolloutGroup 的同步是 policy 版本同步。RolloutWorker 是行为策略推理副本，必须在明确边界上从
ActorGroup 获取新权重，并记录本地 policy version。

LearnerGroup 到 WMEnvWorker 的同步是 world model 与 classifier/reward model 版本同步。WMEnvWorker 使用这些
模型进行环境侧推理，但不在本地训练它们。

同步规范：

- 每次同步都应有方向、组件、源版本、目标版本和结果。
- 同步触发条件应来自配置或显式 loop 逻辑。
- 不同组件的版本不能共用一个模糊计数器。
- 不要让 env step、rollout inference 或 replay sample 隐式触发权重更新。
- checkpoint 应保存足够信息恢复同步后的版本关系。

如果某个实现需要跳过同步或使用 stale 权重，应在 metrics 或 diagnostics 中记录原因。

# Configuration

配置应静态、清晰、可验证。运行时派生值可以在 Runner 或 builder 中计算，但最终行为必须能从 resolved config
和日志解释。

要求：

- Hydra 是配置 source of truth。
- shell 脚本只转发参数，不实现训练分支。
- validation 检查关系，不选择隐藏训练行为。
- optional component 只在配置声明时构建。
- checkpoint-specific 设置跟随 task/checkpoint metadata。
- shape、sidecar、token、chunk、latent 等外部契约从 task 和 artifact metadata 派生。

不推荐：

- 在训练循环中硬编码模型类、dataset 类或 worker 类。
- 用环境变量绕过 placement 或 rendering contract。
- 在 YAML 中复制应由 metadata 派生的维度。
- 在 validation 中写入会影响训练语义的默认值。

配置变更应有测试覆盖，至少证明 Hydra compose 和 validation 行为符合预期。

# Testing

测试应覆盖契约，而不只是覆盖实现路径。大型 RL 系统的测试目标是防止边界漂移、shape 漂移、同步漂移和恢复
语义漂移。

推荐测试层级：

- import test：核心 target、Runner、Worker 和 registry 可加载。
- unit test：dataclass、shape、collation、placement、naming、validation、checkpoint helper。
- tiny smoke：最小配置启动 Runner 并完成短流程。
- GPU smoke：验证 device、precision、FSDP、Ray placement。
- e2e test：真实 LIBERO collect、cotrain、eval 或 resume。
- long-run test：验证稳定性、checkpoint、metrics、memory 和吞吐。

新增或修改以下内容时必须优先考虑测试：

- cotrain message 或 trajectory shape。
- ActorGroup 与 RolloutGroup 的边界。
- LearnerGroup 与 WMEnvWorker 的同步。
- checkpoint bridge 和 resume。
- replay 写入、采样和 sidecar key。
- Hydra config、validation 和 launcher command generation。
- metrics namespace。

测试应尽量小而明确。GPU、Ray、真实环境等昂贵测试应放在 e2e 并用显式 gate 控制。

# Documentation

文档应解释设计意图、职责边界和数据流，再给出少量代码名或配置键帮助定位。不要把源码方法逐个搬进文档。

主文档应回答：

- 为什么这样设计。
- 谁负责什么。
- 数据如何流动。
- 权重如何同步。
- checkpoint 如何恢复。
- 哪些边界不能被破坏。
- 当前事实与目标方案是否一致。

涉及实现边界时，可以附带简短 API contract。API contract 用于固定输入、输出、shape、同步顺序和禁止事项，
不是为了替代源码说明。

文档维护规则：

- `spec/` 是 architecture source of truth。
- `99_manual_notes.md` 的用户手写内容受保护。
- 历史计划只能作为参考，不作为当前 architecture 结论。
- 冲突内容优先写成 `Current vs Target` 或 `Open Questions`。
- 不把目标、现状、实验结果和历史计划混在同一段结论中。

文档应长期可读。不要记录会快速过时的临时命令输出、一次性调试日志或未确认猜测。

# Compatibility

向后兼容是长期系统稳定性的组成部分。兼容不是保留所有旧行为，而是让用户和 artifact 有可理解的迁移路径。

兼容策略：

- 旧配置 key 可以被读取，但主动写出的配置应使用正式新名称。
- 旧 checkpoint 可以通过 adapter 加载，但新 checkpoint 应使用当前契约。
- 旧 metric 如需改名，应在迁移期保留映射或明确记录断点。
- 旧 Runner 或 route 若仍可用，应标明 status 和使用边界。
- 删除旧接口前，应确认 active 配置、测试和文档不再依赖。

不应为了兼容让同一概念长期拥有两个正式名称。兼容层应窄、明确、可删除，并有测试保护。

# Review Checklist

提交重要实现前，至少检查以下问题：

- 新类是否只有一个主要职责。
- 新接口是否有明确输入、输出和副作用。
- 新数据结构是否说明字段、shape 和版本。
- 新同步是否有方向、触发条件和版本记录。
- 新配置是否能通过 Hydra compose 和 validation。
- 新 metric 是否使用稳定 namespace。
- 新 checkpoint 是否能解释组件和恢复语义。
- 新命名是否与正式概念冲突。
- 新文档是否区分目标、现状和历史。
- 新测试是否覆盖了最容易漂移的契约。

如果某项检查无法回答，应先补契约或验证，再扩大实现范围。
