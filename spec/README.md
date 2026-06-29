# DreamerVLA Spec

`spec/` 只保留当前仓库主线架构的最小说明。这里不写历史过程，不复制长计划，也不替代代码和
Hydra 配置。

阅读顺序：

| 文件 | 内容 |
| --- | --- |
| [`00_overview.md`](00_overview.md) | 一页架构总览：入口、包结构、主线流程和核心边界。 |
| [`01_goal.md`](01_goal.md) | 项目目标、运行入口、Runner 生命周期和产物位置。 |
| [`02_naming.md`](02_naming.md) | 当前仓库里的主要组件、角色名和目录职责。 |
| [`03_coding_style.md`](03_coding_style.md) | 实现约束：Hydra、Runner、Worker、数据、checkpoint、metrics。 |
| [`04_complete_loop.md`](04_complete_loop.md) | 主线 `collect -> warmup -> cotrain -> eval` 数据流。 |
| [`05_ray_runtime.md`](05_ray_runtime.md) | Ray/manual cotrain 的 group、worker、placement 和同步边界。 |
| [`99_manual_notes.md`](99_manual_notes.md) | 用户第一性指导，保留原文；扩写架构时再参考。 |

## Source Rule

本目录优先描述当前仓库事实：`dreamervla/`、`configs/`、`scripts/`、`tests/` 中实际存在的入口和边界。
如果 `99_manual_notes.md` 与当前代码不同，主文档应写清当前实现，不把目标方案伪装成已落地事实。

## Normative And Reference Files

当前 architecture source of truth 是上表列出的紧凑主文档，以及最高优先级的
`99_manual_notes.md` 用户第一性指导。

`superpowers/plans/` 中的长计划是 historical/reference implementation plans，不是当前
architecture source of truth。保留它们是为了追溯历史设计与执行上下文；使用前先阅读
[`superpowers/README.md`](superpowers/README.md) 中的边界说明。

[`98_prompt.md`](98_prompt.md) 是 reference-only historical prompt context，不在正常阅读顺序中。

## Keep It Small

新增内容只有在能回答下面问题时才放进 `spec/`：

- 从哪个入口运行？
- 哪个组件负责什么？
- 数据从哪里来、到哪里去？
- checkpoint、metrics、日志落在哪里？
- 需要用什么测试或 smoke 验证？

历史计划、长实现记录和临时调试过程不要放进主目录。
