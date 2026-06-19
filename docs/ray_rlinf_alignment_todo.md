# Ray 对齐:待实现(DreamerVLA → RLinf 剩余 TODO)

- 日期:2026-06-19
- 用途:把 DreamerVLA 可选 Ray backend 对齐 RLinf **尚未完成**的工作汇总成一份单一 TODO。
- 已完成的部分与设计立场见 **`docs/ray_rlinf_alignment_implemented.md`**;对齐基准是相邻 `RLinf` 仓库。

> **现状**:单机核心对齐已完成(scheduler 骨架、workers、真实 learner 闭环、FSDP/AMP/offload/FSDP2 显存栈、
> collective/bucket/patch/压缩权重同步、模型注册表、手动 config groups)。**剩余收敛为**:
> 少数重型/条件项(P3)。**多节点横向扩展不是 DreamerVLA 目标,不再作为待实现项推进。**
>
> 优先级:**P0 = 阻塞训练正确性** · **P1 = 单机扩展** · **P3 = 重型/条件**。

---

## ✅ P0 + P1 —— 已完成(2026-06-19,TDD + 验证;详见 implemented 文档 §1.8)

- **P0 训练等价 parity** — `tests/e2e_tests/test_s5_learner_parity.py`:ray-actor `LearnerWorker` 与
  in-process learner 在同一 fixed batch 上 `rl/actor_loss`/`returns_mean`/`policy_grad_norm` 逐位一致
  (`workers/replay/_test_replays.py:FixedBatchReplay`)。
  **注**:两个 cotrain *runner*(ray vs 单机)循环结构不同(async/sync、offline warmup、cadence),
  全循环聚合 `allclose` 不可行也无意义;真正该守的等价 = **learner 更新数学跨 actor 边界一致**,已守住。
- **P1 collective send/recv + 多通道** — `scheduler/collective/torch_group.py` 的 `send/recv`(`isend`+`flush_sends`,
  channel=tag);真 2-rank gloo e2e `tests/e2e_tests/test_s1b_collective_send_recv.py`。
- **P1 权重同步 bucket/patch/压缩** — `weight_syncer/{bucket,patch,compression}.py`(object-store 背书,可单机验证)。
- **P1 FSDP2 + strategy 子树** — `hybrid_engines/fsdp/strategy/{base,fsdp,fsdp2,checkpoint}.py`,
  `FSDPModelManager.make_strategy()` 委派,新增 `fsdp2`;单机 `WORLD_SIZE=1` passthrough 可验证。
- **P1 config 早校验** — `config.py::_validate_fsdp_config` 对 `learner.train_cfg.fsdp` 的 strategy/precision fail-fast。
  (更大范围 config dataclass 化:**已收窄/暂缓**——`FSDPModelManager` 本就是 `@dataclass`,高价值早校验已补;
  全量 dataclass 化 ROI 低且 `config.py` 在活跃演进,留待需要时再做。)

> 回归覆盖:对应 unit/e2e 已补到仓;最终全量验证以本次提交后的命令输出为准。

---

## P3 —— 重型 / 条件项

### [ ] `scheduler/dynamic_scheduler/` 升级为 RLinf 式调度器
- 把当前 executor-backed 最小 `ComponentScheduler` 升级为 RLinf 式 `ComponentManager` 组件级预取/流水线编排。
- (含原 H3:`cold_start_ray_collect_runner._run_loop_overlap` 升级为 RLinf sync-pipeline parity——推理跑 batch t 时上批 env-step 在飞、预取下帧 obs,仅 dump-size 达标时阻塞;不采用完整 `AsyncEmbodiedRunner` 并发。)

### [x] Channel async API
- `channel.py` 已有 key 路由和 weighted batch;已补统一 `AsyncWork` 句柄、`put_no_wait/get_no_wait`
  以及 batch/weighted batch no-wait 变体。

### [ ] reward/critic worker(条件)
- `workers/{reward,critic}/`:仅当 RL 需独立 reward 服务 / 独立 critic worker 才补(当前 outcome 模式不需要)。

### [ ] hardware 注册表扩展 + 高效 kernel
- `scheduler/hardware.py` 由 CUDA 发现扩成 NPU/机器人注册表(对齐 RLinf `hardware/`)。
- liger_kernel / FlashAttention 可选接,默认关(由 config 显式开启)。

### [ ] 条件:Megatron / vLLM / SGLang
- `hybrid_engines/megatron/`(TP/PP/SP + transformer/model-parallel config 派生 `build_transformer_config`/`_build_model_parallel_config`)——仅当单卡放不下、需大模型并行时。
- vLLM / SGLang 推理引擎——仅当策略变自回归 LLM 时。
- **默认维持非目标**,除非 DreamerVLA 模型规模/形态改变。

---

## 备注

- 多节点不是目标;单机所有 P0/P1/P3 项都只用 loopback + 共享内存(见 implemented 文档 §4)。
- 所有显存/资源项均为**手动杠杆 + 早校验 + 可观测**,**不做 VRAM 自适应**(见 implemented 文档 §3.1)。
- 每补一项加配套测试;真实更新步相关回归优先(P0)。
