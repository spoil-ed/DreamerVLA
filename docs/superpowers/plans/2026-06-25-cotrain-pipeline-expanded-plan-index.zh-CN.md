# 2026-06-25 Cotrain Pipeline Expanded Plan Index

这组文件把
`docs/superpowers/plans/2026-06-25-cotrain-pipeline-decisions-and-execution.zh-CN.md`
拆成多份可独立执行、可测试、可提交的实现计划。执行顺序按 P0/P1/P2 排列。

## 执行顺序

1. `2026-06-25-cotrain-p0-action-chunk-real-metrics.zh-CN.md`
   - 修真实 rollout 的 action chunk 执行契约。
   - 锁定 real completed episode 口径的 success metrics。

2. `2026-06-25-cotrain-p1-replay-ready-and-sampling.zh-CN.md`
   - 补 full-run replay ready gate。
   - 收口三池采样和 latest online 优先。
   - 让 no-Ray 与 Ray replay worker 共享同一 ready 语义。

3. `2026-06-25-cotrain-p1-warmup-replay-epochs.zh-CN.md`
   - 将 WM replay epoch 与 classifier candidate-window epoch 分开计算。
   - 保留 max-step cap。
   - 确认 WM/classifier warmup 在 replay 内交替执行。

4. `2026-06-25-cotrain-p1-task-conditioning.zh-CN.md`
   - 为 multi-task cotrain 增加可关闭的显式 task conditioning。
   - replay batch、WM、classifier 通过 `task_ids` 串起来。
   - 默认单任务路径保持关闭且数值不变。

5. `2026-06-25-cotrain-p2-adaptive-imagination-signals.zh-CN.md`
   - 锁定 bounded adaptive imagined rollout group。
   - 区分 imagined score 与 real success rate。
   - 收口 PPO/LUMOS actor-signal metrics。

6. `2026-06-25-cotrain-p2-metadata-eval-ray-contracts.zh-CN.md`
   - 补 episode-level metadata contract。
   - 接入 periodic real eval 的轻量调度层。
   - 明确 Ray OFT fixed-base 与 Ray learned-actor rollout 的模式边界。

## 执行规则

- 每份计划都要求执行 agent 使用 `superpowers:subagent-driven-development` 或
  `superpowers:executing-plans`。
- 每份计划都先写 CPU 单测，再改实现，再跑定向测试。
- GPU/LIBERO full-run 命令只作为最终验收，不替代 CPU 单测。
- 如果某个步骤中的测试在当前工作区已经通过，执行 agent 应保留已有实现，继续后续验证和提交。
