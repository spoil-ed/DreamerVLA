# Historical Prompt Reference

This file is historical/reference prompt context.

It is not current architecture source of truth. Use it only to understand earlier task framing; current architecture facts live in `spec/README.md`, `00_overview.md` through `05_ray_runtime.md`, and the highest-priority user guidance in `99_manual_notes.md`.

---

实现 [明确目标]；
完成标准是 [测试/构建/指标/产物] 通过；
不要修改 [边界/公共 API/数据库 schema]；
每个 checkpoint 后运行 [验证命令]；达成标准后停止并总结；
如果被阻塞，报告证据和需要我补充的资料、验证信息或者真机测试结果。


可以补充一个「参考实现与学习要求」章节，同样保持忠实于你的原意：

## 补充要求

### 文档说明

1. 所有相关文档已经迁移至：

   `/mnt/data/spoil/workspace/DreamerVLA/spec`

2. 你可以对 `99_manual_notes.md` 进行少量修改，但修改内容**仅限于统一和对齐仓库中的命名**，以避免当前名称不一致带来的混乱。不要修改其中的设计逻辑、流程或其他内容。
3. `/mnt/data/spoil/workspace/DreamerVLA/docs/superpowers/plans/2026-06-28-manual-cotrain-rlinf-alignment.zh-CN.md`内为已有的方案，请你阅读，并分子代理做好详细的计划
---

## 执行策略

1. 请先使用 superpowers 制定一份完整、详细的实现方案和计划（plan）。
2. 详细阅读已有的方案，开始后续实现。
3. 方案通过后，请基于该方案启动子代理，并按照以下 loop 推进：

   ```
   对比方案与当前代码实现
   -> 修改代码
   -> 执行 debug / smoke test
   -> 根据结果继续修正

   ```
4. 不要在缺少整体方案、也没有经过审查的情况下直接开始编码。
5. 后续所有实现和 debug 都必须围绕已审查通过的方案进行，避免实现过程中偏离主线。

---

## 参考实现与学习要求

1. 在实现过程中，代码组织方式、具体实现细节以及数据流设计可以参考：

   `/mnt/data/spoil/workspace/RLinf`

   尤其是在 Group、Worker、数据流以及训练流程组织等方面，应优先参考 RLinf 中已经验证可行的方案。

2. 请充分学习并理解当前仓库和 RLinf 中关于 **Group** 的设计与使用方式，在开始实现前确保已经掌握相关机制。

3. 建议先尝试完整跑通 RLinf 中的 **embodied** 方案。只有真正理解该方案的运行流程、组件关系以及启动方式后，才能更准确地实现 DreamerVLA 的 cotrain 流程。

---

## 最终目标

本次任务的最终目标是：**能够成功启动 cotrain 流程。**

所有 architecture 相关文档均以：

`/mnt/data/spoil/workspace/DreamerVLA/spec`

中的内容为准。

最终实现必须严格按照：

`/mnt/data/spoil/workspace/DreamerVLA/spec/99_manual_notes.md`
`/mnt/data/spoil/workspace/DreamerVLA/docs/superpowers/plans/2026-06-28-manual-cotrain-rlinf-alignment.zh-CN.md`中的计划实现中的说明来启动 cotrain。

---

## 完成标准

只有满足以下所有条件，本次任务才算完成：

* cotrain 流程能够成功启动；
* 启动流程严格遵循 `99_manual_notes.md` 中的说明；
* 支持使用 **0–5 张 GPU** 启动训练；
* 能够进行长时间训练，只需要能够成功启动并完成一次完整的global_step即可；
* 实现过程中应优先保证与当前仓库的兼容性，而不是引入新的设计。
* 完整实现 `/mnt/data/spoil/workspace/DreamerVLA/docs/superpowers/plans/2026-06-28-manual-cotrain-rlinf-alignment.zh-CN.md` 内的计划。
以上是本次任务的核心目标，请始终围绕该目标推进实现。
