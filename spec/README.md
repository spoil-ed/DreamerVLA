# DreamerVLA Spec

`spec/` 是 DreamerVLA 当前方案的说明入口。这里用于解释项目的主线目标、
architecture 判断、训练流程、关键边界、当前实现状态和后续待补内容。

这个目录的目标是：只阅读 `spec/`，就能理解 DreamerVLA 当前为什么这样设计、
应该如何运行、哪些部分已经确定、哪些部分仍需要验证。

## Contents

- `00_overview.md`：主线架构入口，说明整体目标、核心 group、训练形态和重要边界。
- `01_complete_loop.md`：端到端流程说明，从 collect、warmup 到 cotrain 和 eval。
- `02_ray.md`：Ray/manual cotrain 的 group、worker、channel、placement 和同步说明。
- `03_current_implementation.md`：当前代码事实，区分已实现能力和仍需运行验证的内容。
- `04_rlinf_alignment.md`：DreamerVLA manual cotrain 与 RLinf/WoVR group、worker、channel
  组织方式的对齐说明。
- `05_cotrain_data_contracts.md`：Env/Rollout/Actor/Learner/replay/WMEnv 之间的数据结构、
  shape 和 sidecar 契约。
- `06_sync_checkpoint_metrics.md`：权重同步、warmup checkpoint bridge、manual checkpoint
  和 metrics namespace。
- `07_validation_matrix.md`：unit、tiny smoke、GPU/LIBERO e2e 和 long-run 验证矩阵。
- `98_prompt.md`：任务提示和迁移背景，仅作历史参考。
- `99_manual_notes.md`：用户第一性指导，优先级最高；除非用户明确要求，不得移动、压缩、
  重写或删除其中的用户手写内容。
- `superpowers/plans/`：执行计划和实现过程记录，可作历史参考，但不作为当前
  architecture source of truth。

## Writing Style

本目录的主文档应以自然语言描述为主。先解释“为什么这样设计、谁负责什么、数据如何流动、
哪些边界不能破坏”，再用少量代码名、配置键或命令辅助定位。

涉及实现边界时，可以在自然语言之后加入带注释的 API 描述。API 描述用于固定契约，
不是为了替代方案说明，也不应该把源码方法逐个搬进文档。

推荐形态：

```markdown
## Component Or Flow

先用自然语言说明设计意图、职责边界、数据流和取舍。

### API Contract

用带注释的伪接口说明输入、输出、shape、同步顺序和禁止事项。
```

具体 API contract 示例、shape 约束、资源配置、数据 artifact、验证矩阵和待填写约束，
应写入对应正文文档，例如 `03_current_implementation.md`、`05_cotrain_data_contracts.md`、
`06_sync_checkpoint_metrics.md` 或 `07_validation_matrix.md`。

## Source Priority

当文档之间出现冲突时，按以下顺序处理：

1. 以 `99_manual_notes.md` 的第一性判断为最高优先级。
2. 检查当前代码和测试是否已经形成新的事实。
3. 将差异明确写成 `Current vs Target` 或 `Open Questions`。
4. 确认后再更新主文档，不要把目标、现状和历史计划混写在一起。

## User Constraints

本节预留给用户后续填写新的 architecture、实现、运行或验证约束。这里的内容是后续主文档
展开的输入来源，不要求一开始完整。

AI agent 不应在没有用户明确要求时主动改写、归并或删除本节内容。需要整理时，应先保留
用户原始约束，再在 `00_overview.md`、`01_complete_loop.md` 或 `02_ray.md` 等正文文档中展开。

待用户填写。
