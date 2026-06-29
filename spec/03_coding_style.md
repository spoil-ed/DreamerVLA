# Implementation Rules

本文件只保留影响架构边界的实现规则。

## Hydra

- Hydra 是配置 source of truth。
- 模型、Runner、dataset、logger、precision、parallelism、checkpoint path 都应来自 config。
- `validate_cfg` 只校验关系，不偷偷选择训练行为。
- 新路线优先增加 config group 和 experiment recipe，不在训练 loop 里写 `if model == ...`。

## Runner And Worker

- Runner 负责 job 生命周期、日志、checkpoint、metrics 和阶段调度。
- Worker 只负责一个 runtime 角色：env step、rollout inference、actor training、learner update 或 replay。
- 脚本保持薄封装；循环、分支、资源规划进入 Python/Hydra。
- 可选组件必须 opt-in，不要在公共 loop 里硬失败。

## Data

- 跨 worker 数据使用结构化 message、manifest 或 dataclass。
- 关键字段必须能说明 `task_id`、`episode_id`、`slot_id`、`global_step`、版本和 shape。
- OpenVLA-OFT 的 downstream shape 从 task metadata、sidecar metadata 和 collected artifact 派生。
- 不用裸 `dict[str, Any]` 传递核心训练契约，除非边界已经有 schema。

## Checkpoint And Metrics

- checkpoint 要说明组件、global step、版本和 resume 语义。
- base checkpoint 使用 Runner helper；pipeline warmup checkpoint 放在 cotrain 子目录。
- metrics namespace 保持稳定：`env/`、`rollout/`、`actor/`、`train/`、`eval/`、`sync/`、
  `replay_buffer/`、`time/`。
- training loop 中避免裸 `print`，优先 runner logging、JSON logger、TensorBoard、W&B。

## Verification

按风险选择验证层级：

1. import check
2. unit test
3. tiny smoke
4. GPU/Ray smoke
5. LIBERO e2e
6. resume/long-run stability

改动 shared config、checkpoint、worker message、trajectory shape、FSDP/Ray placement 时，至少要有对应测试或 smoke。
