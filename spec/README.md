# DreamerVLA Spec

`spec/` 是 DreamerVLA 当前方案的说明入口。这里用于解释项目的主线目标、
architecture 判断、训练流程、关键边界、当前实现状态和后续待补内容。

这个目录的目标是：只阅读 `spec/`，就能理解 DreamerVLA 当前为什么这样设计、
应该如何运行、哪些部分已经确定、哪些部分仍需要验证。

`spec/` 是 DreamerVLA 的 architecture source of truth。

代码应逐步向文档收敛，而不是让文档被代码实现细节牵着走。

---

# Contents

* `00_overview.md`：主线架构入口，说明整体目标、核心 group、训练形态和重要边界。
* `01_goal.md`：项目目标、长期 architecture 原则和主线边界。
* `02_naming.md`：正式命名、模块命名、worker 命名和兼容命名规则。
* `03_coding_style.md`：面向长期 RL 系统维护的编码风格、接口设计和结构化数据规则。
* `04_complete_loop.md`：端到端流程说明，从 collect、warmup 到 manual cotrain 和 eval。
* `98_prompt.md`：任务提示和迁移背景，仅作历史参考。
* `99_manual_notes.md`：用户第一性指导，优先级最高；除非用户明确要求，不得移动、压缩、重写或删除其中的用户手写内容。
* `superpowers/plans/`：执行计划和实现过程记录，可作历史参考，但不作为当前 architecture source of truth。

---

# Writing Style

本目录的主文档应以自然语言描述为主。

先解释：

* 为什么这样设计；
* 谁负责什么；
* 数据如何流动；
* 为什么这样划分边界；
* 哪些边界不能被破坏；

然后再使用少量代码名、配置键或命令辅助定位。

涉及实现边界时，可以在自然语言之后加入带注释的 API 描述。

API 描述用于固定契约，不是为了替代方案说明，也不应该把源码方法逐个搬进文档。

推荐形态：

```markdown
## Component Or Flow

先用自然语言说明设计意图、职责边界、数据流和取舍。

### API Contract

用带注释的伪接口说明输入、输出、shape、同步顺序和禁止事项。
```

具体 API contract、shape 约束、资源配置、数据 artifact、验证矩阵和待填写约束，
应写入对应正文文档，而不是集中堆积到单一文档中。

---

# Architecture Principles

DreamerVLA 的 architecture 应长期遵循以下原则。

## Single Responsibility

每个组件只负责一个主要职责，例如环境交互、策略推理、参数更新、数据存储与采样、
评估与指标统计应分属不同组件。

避免单个组件同时承担：

* collect；
* learning；
* checkpoint；
* logging；
* evaluation；
* synchronization；

等多个核心职责。

具体组件及其职责对应关系见 `01_goal.md`、`02_naming.md` 与 `04_complete_loop.md`。

## Explicit Over Implicit

系统行为应尽可能显式。

避免：

* 隐式同步；
* 隐式状态切换；
* 隐式权重更新；
* 隐式资源占用；
* 隐式副作用。

所有关键状态变化都应能够被日志追踪。

## Stable Public Interface

公共接口应保持长期稳定。

优先增加兼容层，而不是直接删除旧接口。

## Minimal Abstraction

除非能够显著降低复杂度，否则不要新增抽象层。

优先重构，而不是重写。

---

# Naming Principles

同一个概念只能存在一个正式名称，不要为同一角色引入多个并行名称或随意追加版本后缀。

正式名称及其对应关系以 `02_naming.md` 与 `04_complete_loop.md` 为准。

废弃名称应保留在 compatibility 层，并明确标记 deprecated。

---

# Runtime Invariants

以下约束默认始终成立。

* collect 可以与 learner update 并行执行；
* replay artifact 必须支持 resume；
* actor、rollout、world model 权重必须具备版本概念；
* checkpoint 必须支持完整恢复训练；
* metrics namespace 应保持稳定；
* 所有关键数据路径必须可追踪；
* 任何核心数据都必须能够说明来源、流向和落盘位置。

如果实现违反上述约束，应明确记录原因。

---

# Validation Philosophy

所有重要修改都应经过验证。

推荐验证层级：

1. import check；
2. unit test；
3. tiny smoke；
4. GPU smoke；
5. LIBERO end-to-end；
6. long-run stability。

新增功能或大规模重构后，不应只依赖静态阅读判断正确性。

---

# Source Priority

当文档之间出现冲突时，按以下顺序处理：

1. `99_manual_notes.md` 的第一性判断。
2. 当前代码与测试已经形成的事实。
3. 当前主文档中的明确 architecture。
4. 历史计划和旧方案。

发现冲突时：

* 不要直接覆盖历史内容；
* 优先写成 `Current vs Target`；
* 或写成 `Open Questions`；
* 待确认后再更新主文档。

禁止把目标、现状和历史计划混写。

---

# Refactor Rules

进行重构时：

1. 优先保持行为一致；
2. 优先增量修改；
3. 避免大规模重写；
4. 每次修改后执行验证；
5. 所有重要修改都应记录；
6. 不得删除用户手写内容；
7. 优先 deprecated，而不是 remove。

---

# User Constraints

本节预留给用户后续填写新的 architecture、实现、运行或验证约束。

这里的内容是后续主文档展开的输入来源，不要求一开始完整。

AI agent 不应在没有用户明确要求时主动改写、归并或删除本节内容。

需要整理时，应先保留用户原始约束，再在正文文档中展开。

待用户填写。
