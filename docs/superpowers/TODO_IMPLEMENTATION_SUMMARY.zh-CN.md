# TODO Implementation Summary

> 本文是 `TODO.md` 的执行摘要,用于快速判断后续需要实现哪些任务。
> 任务细节、验收条件和归档规则以 [`TODO.md`](TODO.md) 为准;已完成项归档到
> [`../HISTORY.md`](../HISTORY.md)。当本文与 `TODO.md` 不一致时,以 `TODO.md` 为准。

- Last updated: 2026-06-27
- Scope: 当前仍未归档的实现任务,不包含 non-goal / won't-fix 项。
- Done rule: 任务必须完成实现、接入主路径、通过相应验证,然后从 `TODO.md` 移到
  `HISTORY.md`。

## Execution Order

1. **先稳定 checkpoint / DDP / RNG**
   - `RUN-01 smoke`
   - `X-01②`
   - `RLINF-01 remainder`

   这三项互相依赖:先证明 RynnVLA 多 GPU save/resume 路径可跑,再统一
   `online_dreamervla` checkpoint envelope,最后做 bit-exact RNG 验证。

2. **补齐 Ray/GPU 观测与真实长跑**
   - `RLINF-02 remainder`
   - `offline-warmup -> online-cotrain real long-run`
   - `Perf benchmark`

   目标是把分散的 `time/...` 指标统一到 `Timers` / `Profiler`,并产出真实
   LIBERO/OFT cotrain 长跑指标和吞吐/显存基准。

3. **做 Hydra 解耦**
   - `DECOUPLE-02`
   - `DECOUPLE-03`
   - `DECOUPLE-04`

   目标是消除直接实例化、runtime `_target_` mutation 和模型/数据/实现类硬耦合,
   让组件选择回到 Hydra source-of-truth。

4. **做性能与内存优化**
   - `Perf W7`
   - `Perf H7`
   - `Perf H2`
   - `Perf W5`
   - `Perf W2 caller-wiring`
   - `Perf W8`
   - `Perf H3`
   - `Perf H6`

   这些任务分为数据加载、world model 训练精度/缓存、replay 布局、OFT 解码、
   checkpoint caller wiring 和 frozen eval-only 路径优化。具体实现口径见 `TODO.md`。

5. **最后做结构性重构**
   - `MEM-RL-01 remainder + MEM-RL-02`
   - `online_dreamervla.main() split`

   这两项影响训练循环结构,应在 checkpoint/DDP/save-load 区域稳定后再推进。

## Task Groups

### GPU-Gated

需要 GPU/LIBERO E2E 或真实 checkpoint 验证:

- `RUN-01 smoke`: RynnVLA 多 GPU save/resume smoke。
- `X-01②`: checkpoint 格式统一到 BaseRunner envelope。
- `RLINF-01 remainder`: 多 GPU save/resume bit-exact RNG 验证。
- `RLINF-02 remainder`: Ray/GPU loop timing 统一接入 `Timers` / `Profiler`。
- `DECOUPLE-02`: 关键组件通过 Hydra `instantiate(cfg.<x>)` 构建。
- `DECOUPLE-03`: `L1RegressionActionHead` 通过协议/config 注入。
- `Perf W7/H7/H2/W5/W2`: GPU 相关性能与 checkpoint caller wiring。
- `offline-warmup -> online-cotrain real long-run`: 真实 OFT/LIBERO cotrain 长跑。
- `Perf benchmark`: 基于真实运行的吞吐/显存 benchmark。

入口: [`TODO.md#gpu-gated-needs-a-gpu-box--libero-e2e`](TODO.md#gpu-gated-needs-a-gpu-box--libero-e2e)

### CPU-Doable

可先在本机通过单测推进:

- `DECOUPLE-04`: 小型 Hydra 解耦和 shared util 清理。
- `Perf W8`: frozen eval-only bf16 + `inference_mode`。
- `Perf H3`: replay readiness 增量计数。
- `Perf H6`: WM KV-cache。

入口: [`TODO.md#cpu-doable-next-unit-testable-here`](TODO.md#cpu-doable-next-unit-testable-here)

### Structural

结构性重构,需要排在核心训练/存档路径稳定之后:

- `MEM-RL-01 remainder + MEM-RL-02`: 显式 imagination host buffer + WM-as-env。
- `online_dreamervla.main() split`: 拆分 `parse_args` 和大 `main()` loop。

入口: [`TODO.md#structural-refactors`](TODO.md#structural-refactors)

### Ray Backend Remaining

Ray 主线剩余只保留两类:

- 真实 LIBERO/OFT 长跑。
- benchmark-driven performance tuning。

其他 P3 项保持 trigger-only:独立 reward/critic worker、hardware registry/kernel
switch、Megatron/vLLM/SGLang 只有在需求触发时才做。

入口: [`TODO.md#ray-backend-remaining-ray_rlinf_alignment_implementedmd-is-the-shipped-record`](TODO.md#ray-backend-remaining-ray_rlinf_alignment_implementedmd-is-the-shipped-record)

## Non-Goals

不要把以下内容重新打开为实现任务:

- multi-node horizontal scaling
- VRAM auto-sizing / auto-batch / OOM-retry
- collocated / disaggregated / hybrid placement modes
- channel key-routing as a target
- `TODO.md` 的 won't-fix / intentional 列表

入口: [`TODO.md#non-goals--do-not-pursue`](TODO.md#non-goals--do-not-pursue)
