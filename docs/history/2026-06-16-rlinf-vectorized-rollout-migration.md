# 并行采集器·卡内向量化迁移(参考 RLinf libero_env)

> **归档说明(2026-06-23)**:本设计稿已从 `docs/superpowers/specs/` 移入设计史;所述卡内向量化采集已实现。
> 原 §10 的"未来 Ray 方案"已落地为可选 Ray backend,以 `docs/ray_*.md` 为准。

- 日期:2026-06-16
- 状态:设计/迁移方案,待确认后写实现计划
- 范围:**只针对 plan 的 Task 7 Step 2–3**(卡内 K-env 真并行 + batched VLA 推理)。
  跨卡分片(Layer 1, torchrun M-rank)已实现,不在本次改动范围。
- 相关文档:
  - 原始设计:`docs/history/2026-06-16-parallel-rollout-collection-design.md`
- 相关代码(本仓):
  - `dreamervla/runners/collect_parallel_rollouts.py`(采集器入口)
  - `dreamervla/runners/rollout_hidden_extractor.py`(`OFTRolloutHiddenExtractor`,当前单 obs)
  - `dreamervla/envs/train_env.py`(`full_record()` 已就绪)
  - `dreamervla/dataset/rollout_dump_writer.py`(`RolloutDumpWriter`)
- 参考实现(RLinf):
  - `RLinf/rlinf/envs/venv/venv.py`(`BaseVectorEnv` / `SubprocEnvWorker` 通用向量化框架)
  - `RLinf/rlinf/envs/libero/venv.py`(`ReconfigureSubprocEnv`,LIBERO 任务重配)
  - `RLinf/rlinf/envs/libero/libero_env.py`(`LiberoEnv` 批量 obs 包装 / `chunk_step` / auto-reset)
  - `RLinf/rlinf/workers/env/env_worker.py`(高层 rollout 循环)

---

## 1. 目标与范围

把"卡内 K 个 env 真并行步进 + 一次批量前向推理"补上,使单卡吞吐随 K 提升,
而不是当前(被删的)实验版那样付了 IPC 开销却≈K=1。

- **范围内**:`OFTRolloutHiddenExtractor` 的批量推理路径、一个最小向量化 env wrapper、
  采集主循环改成连续步进 + auto-reset。
- **范围外**:Layer 1 跨卡分片(已实现,不动)、落盘 schema/侧车/分类器数据集(已实现,不动)、
  RynnVLA-legacy 路由(后续按 `expected_*` 切,不在本次)。

---

## 2. 现状(对照 plan 的 Task 1–9)

| 阶段 | 交付物 | 状态 |
|---|---|---|
| P1 | env `full_record` 模式(`train_env.py:277`) | ✅ |
| P1 | `RolloutDumpWriter`(`rollout_dump_writer.py`) | ✅ |
| P1 | `OFTRolloutHiddenExtractor`(`rollout_hidden_extractor.py`) | ✅ **仅单 obs 推理(batch=1)** |
| P1 | 单进程采集器 | ✅(commit e10eb7c) |
| P1 | proprio/state 字段映射探针 | ✅ `scripts/probe_field_mapping.py`(未 track) |
| P2 | `CollectedRolloutClassifierDataset` | ✅ |
| P3 | Task7-1 torchrun M-rank 跨卡分片(Layer 1) | ✅(commit 447b8c5) |
| **P3** | **Task7-2/3 卡内 K-env 并行 + batched 推理** | ❌ **本次缺口** |
| P3 | Task7-4 显存 fence(`set_per_process_memory_fraction`) | ✅ |
| P4 | `configs/experiment/collect_rollouts_action_hidden.yaml` | ❌ 缺失 |
| P4 | `scripts/run_collect_rollouts.sh` | ❌ 缺失 |
| P5 | 教程"冷启动并行采集"一节 | ❌ 未加 |

> **工作区当前坏掉的中间态**:`git diff` 显示本次改动删了 266 行(整段实验性 Layer-2:
> `_subproc_worker` / `SubprocEnvHandle` / `_try_launch_subproc_envs` / `_run_episode_subproc`),
> 但 `main()` 的调用点仍在(`collect_parallel_rollouts.py:490, 503, 522, 565`),引用已不存在的符号。
> 后果:`envs_per_gpu=1`(默认)单进程 + torchrun 跨卡仍可跑;`envs_per_gpu>1` 直接 `NameError`。
> **本次迁移第 0 步 = 清掉这些残留调用点,让工作区回到可跑的 Layer-1 状态**,再在干净基线上重建。

---

## 3. 被删的 Layer-2 为何不算并行

残留的 `main()` 编排(`collect_parallel_rollouts.py:559-581`)实质是:

```text
for handle in batch:          # 一个一个 handle
    run_full_episode(handle)  # 跑完整条 episode 才换下一个 handle
```

注释自承 `"sequential inference is the ACCEPTABLE fallback"`。两个致命点:

1. **env 串行**:`send → 等 recv → 下一个`,K 个子进程从不同时 step。
2. **推理串行**:每个 obs 单独前向,K 个 obs = K 次前向。

净效果:K 个子进程的 IPC/内存全付了,吞吐≈K=1。这是删它的原因,也是不能简单复活的原因——
**问题不在 Pipe 脚手架,在编排与推理粒度**。

---

## 4. RLinf 的并行机制(三要点)

1. **真·向量化 step**(`venv.py: BaseVectorEnv.step`):
   **先把 K 个 action 全部 `send`,再统一 `recv`**——K 个 env 真正并发步进;
   结果 `np.stack` 成 `[num_envs, ...]` 批。
   > 对比被删版的"send 一个就等 recv",这是唯一让 env 并发起来的关键。

2. **批量喂模型**(`libero_env.py: _wrap_obs`):返回
   `{main_images:[N,H,W,C], wrist_images:[N,...], states:[N,...]}`,
   policy **一次前向吃 N 个 obs**。

3. **连续步进 + auto-reset**(`libero_env.py: chunk_step` / `_handle_auto_reset`):
   不是"一条 episode 跑完再下一条",而是所有 env 同步步进,谁 `done` 就**只 reset 那一个**
   (`env_idx = arange(num_envs)[dones]`),`final_obs` 塞进 `info`;
   任务/init_state 用 `reset_state_ids` 预分配,可经 IPC `reconfigure_env_fns` 在运行时换任务
   (`ReconfigureSubprocEnv`)。

一句话:**send-all-then-recv-all(env 并发) + 一次批量前向(GPU 利用率) + done-mask 局部 reset**。

---

## 5. 迁移方案

### 5.0 风险先行验证(决定整件事可不可行)

**OFT `predict_action` 能否 batch>1?** 必须先单独验证再动其余代码。

- 模型架构是 batch-aware 的:`modeling_prismatic.py:998` 用 `input_embeddings.shape[0]` reshape,
  L1-regression 路径(本项目用的就是它,`use_l1_regression=True`)理论支持 B=K。
- 但 **diffusion 路径硬编码 `size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM)`**(`:1026`)——我们不走 diffusion,
  大概率不受影响;仍需核 processor/prompt 拼接、proprio、`_unnormalize_actions` 的 batch 形状。

> **验证 smoke(第一步必做)**:同一个 obs 复制 2 份做一次 B=2 前向,断言
> `actions[0]==actions[1]`、`actions_hidden_states[0]==[1]`,且与单 obs 前向逐元素一致。
> 这条不过,后面所有改造都无意义。

> **✅ 已验证(2026-06-17,GREEN)**:`scripts/smoke_oft_batched_forward.py`。结论:
> - 朴素 `predict_action(B=2)` 在 `modeling_prismatic.py:972`(token `[1,1]` cat)崩溃 —
>   确认 blocker;另有 `:924` 的 `reshape(8,7)` 丢 batch。其余 L1 路径 batch-safe。
> - **绕开 wrapper、直调 batch-safe 内部**(batched token-append + `(B,8,7)` reshape)后:
>   B=1 与 `extractor.step` **逐位相等**;decoded action **partner-invariant(drift=0)**,无跨样本泄漏;
>   `obs_embedding` 批量 vs 单条残差 ~0.23–0.34(fp16 max-abs),与现有 TF-vs-PIL gold 容差 ~0.25 同量级。
> - **设计修正**:批量 `obs_embedding` 与单条**不逐字节一致**(bf16 批量核非确定性)→ §8 验收改为容差判定。

> **✅ 头模式修正(2026-06-17)**:冷启动 base = **one-trajectory OFT ckpt**
> (`Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1`),是 **DISCRETE / 无 L1 头**——
> action 由 **LM logits→argmax→bin_centers** 解码,**不经 action head**(和 RLinf rlinf 版同构)。
> `obs_embedding`(action-query hidden)**与头无关**,L1/discrete 取同一层,故落盘产物不变;
> 只有"解码出 action 去 step env"这一步分流。已实现:`_batched_predict` 按 `action_head is None`
> 分 L1 / discrete(都修掉上游 batch=1 reshape);采集器 `resolve_oft_policy_mode` auto-detect,
> discrete→`use_l1=False`/`use_proprio=False`/`action_head_type=oft_discrete_token`/`include_state=False`
> (对齐 preprocess)。测试 `test_batched_forward_discrete_headless`(B=1 逐位相等、partner-invariant)
> + 1 卡集成 smoke(默认 ckpt 即 discrete,产物 schema/config 全对)。L1 路径保持逐位不变。

### 5.1 批量 extractor

给 `OFTRolloutHiddenExtractor` 加 `step_batch(obs_list, task_descriptions) -> (action_chunks, flat_hiddens)`:

- 维护 **K 个独立 history buffer**(或沿用被删版做法:持有 K 个 extractor 实例,history 天然隔离)。
- 把 K 个样本堆成 `pixel_values[K, num_views*..., ...]`,批量 `input_ids` / `attention_mask` / `proprio`。
- **一次 `model.predict_action`**(B=K),`actions_hidden_states[K,56,4096]` 切成 K×`(229376,)` float16。
- 单 obs 的 `step()` 保留为 `step_batch` 的 K=1 特例(向后兼容)。

### 5.2 最小向量化 env wrapper

复刻 RLinf 的 `SubprocEnvWorker` 思路(被删的 `_subproc_worker` Pipe 脚手架可作起点),但**改编排为 send-all/recv-all**:

- 子进程每步把本仓现成的 `env.full_record()`(写盘要的完整 schema)回传父进程。
- **IPC 量**:每步 2 视图 × 256×256×3 uint8 ≈ 384KB/env/step,K=8 时每步约 3MB——
  先用 **pipe 跑通**;若成瓶颈再上 RLinf 那套 shared-memory buffer(`venv.py` 的 `_setup_buf`/`ShArray`)。
- spawn 启动(LIBERO/mujoco 句柄须在子进程内创建);每 worker 独立 seed / task / init_state。

### 5.3 采集主循环改成连续步进模型

从"一次一条 episode"改为 RLinf 式连续向量化:

```text
K 个 env slot;work-list=(task_id, episode) 跨 slot 分配
loop:
  obs_batch        = gather K 个 full_record()          # recv-all
  action_chunks, H = extractor.step_batch(obs_batch)    # 一次批量前向
  vec_env.step(每 slot 取 chunk[0])                      # scatter + send-all
  各 slot 把当前帧追加到自己"在写的轨迹"
  对 done 的 slot:finalize + writer.write_demo;取下一个 work item 填回(必要时 reconfigure task)
直到 work-list 耗尽且所有 slot 收尾
```

Layer-1 rank 分片(`_shard_work`)与落盘文件名 `r{rank}_shard_*` 不变,套在最外层。

---

## 6. 关键取舍(需拍板)

1. ~~**批量内是否同任务**~~ ✅ **已解决(mixed-task batching)**:`batched_forward` 现在对不同 prompt
   长度做 **左 pad + attention_mask + position_ids=cumsum(mask)-1 + BOS 归位**(对齐 RLinf
   `PrismaticProcessor`),被 pad 的样本与单独算**逐位等价**(块对角注意力,样本独立)。等长 batch
   pad 成 no-op → 原同任务路径仍逐位不变。主循环改为**跨任务单条连续循环**,任务边界处自然出现
   mixed batch,**无 barrier、无尾部空转**;work-list 保持 task-major,slot 只在跨任务时 reopen env。
   实测:`test_batched_forward_mixed_task_matches_single`(短/长 prompt 同批,各自容差内、partner-invariant)。
2. **IPC**:pipe-first(简单) vs shared-memory(快)。**建议** pipe 先通,profile 后再优化。
3. **收集模型**:episode-at-a-time → 连续步进 + auto-reset 是**必要**改动(否则无法自然批量)。
   代价是轨迹切分逻辑从"循环内"挪到"按 done 切片"。

---

## 7. 落地顺序(每步可独立验证)

1. ✅ **第 0 步**:清掉 `main()` 残留 Layer-2 调用点,回到可跑的 Layer-1 基线。
   `collect_parallel_rollouts.py` 编译干净、`envs_per_gpu=1` 端到端通。
2. ✅ **batched-forward smoke**(§5.0)→ GREEN,`scripts/smoke_oft_batched_forward.py`(见 §5.0 实测)。
3. ✅ **`batched_forward`**(§5.1):重构 extractor 抽出 `prepare()` + 模块函数
   `batched_forward()`/`_batched_predict()`,`step` 改为共用一条路径。
   测试(`test_rollout_hidden_extractor.py`):7 纯逻辑 + 3 真模型(B=1 逐位相等;
   K=2 action ≤0.0044、obs_embedding 容差内、partner_drift=0);**回归门**
   `test_inline_matches_offline_sidecar` 仍过(无漂移)。
4. ✅ **向量化 env wrapper**(§5.2):`dreamervla/runners/vec_rollout_env.py`(`VecRolloutEnv`,
   spawn,send-all/recv-all,子集 reset/step)。测试 `test_vec_rollout_env.py`(6 过,fake env)
   + spawn×真 LIBERO 手验(K=2 reset/step,init_state 各异)。
5. ✅ **连续步进主循环**(§5.3):`dreamervla/runners/vectorized_collect.py`(`collect_vectorized`,
   **按任务分批**保证可批量)。测试 `test_vectorized_collect.py`(5 过,fake)。
   接入 `collect_parallel_rollouts._collect_vectorized_path`(`envs_per_gpu>1` 分支)。
   - **1 卡 K=2 集成 smoke**:2 demo,落盘 schema 全对(states (T,79)、obs_embedding
     (T,229376)f16 全非零、init_state 各异、preprocess_config 完整),GPU 14.4GB。
   - **2 卡 K=2 集成 smoke**(`scripts/run_collect_rollouts.sh`):torchrun M=2
     (rank0→cuda:0, rank1→cuda:1),total_work=4 按 rank 分片 2/rank,
     shard 文件 `r0_/r1_` 不冲突,4 demo 覆盖全 (task,ep),**单卡 14.4GB ≤ 80%**。
6. ✅ **P4 launcher**:`scripts/run_collect_rollouts.sh`(瘦 torchrun M-rank 启动器,
   转发 key=value)。
   - ⏳ 待办(非本 md 核心):**严谨吞吐基准**(大 N 摊薄 spawn/load,K=1 vs K=8);
     P4 Hydra experiment YAML(注:采集器是 key=value 独立 runner,非 Hydra 路由 —— 需确认是否要硬接);
     P5 教程"冷启动并行采集"一节。

---

## 8. 验收标准(承接 spec §9 第 1 条)

1. `envs_per_gpu=K>1` 下卡内 K env 并发步进、一次批量前向;单卡显存 ≤ 80%,吞吐 **显著高于** K=1。
2. 批量产出与逐条产出**容差一致**(非逐字节 —— bf16 批量核非确定性,见 §5.0):
   `states`/`images`/`actions` 由 env 决定,逐字节对齐;`obs_embedding` 容差对齐
   (max-abs ≤ 0.5 且 pearson > 0.999,同现有 sidecar 一致性门限)。保证并行只改吞吐、不改数据契约。
3. 产出目录仍被 `train_wm.sh experiment=oft_world_model_dinowm_chunk` **零改动**消费(回归 spec §9-2)。

---

## 9. 风险

- **OFT 批量推理的隐藏 batch=1 假设**(§5.0)——最高风险,先验证。
- **SubprocVecEnv × LIBERO/mujoco**:子进程内 env 创建、spawn 启动、per-worker seed/task/init_state。
- **批量内不同任务的 prompt 变长**:需 padding + attention_mask,否则只能同任务成块(见 §6-1)。
- **IPC 带宽**:full_record 每步回传图像;K 大时 pipe 可能成瓶颈,预留 shared-memory 退路。

---

## 10. 后续 TODO(本期不做,已记录)

- **IPC 共享内存**(`vec_rollout_env.py` 已留 `TODO(perf)`):pipe → RLinf `ShmemVectorEnv`
  风格共享内存 buffer。仅当吞吐 profile 显示 IPC 占比高才上;难点是 `states`/`init_state`
  的场景维 S 随任务变(连续循环跨任务),需 max-S padding 或"定长大图走共享内存 + 变长小字段走 pipe"的混合。
- **Ray(整体 online loop 扩展选项)**:此处原列的"未来 Ray 架构选项"(异构 worker 放置、
  infer-step-learner 流水线重叠、多节点 replay)已落地为可选 Ray backend,详见
  `docs/ray_rlinf_alignment_implemented.md` 与 `docs/ray_online_cotrain_backend.md`(本草案不再重复)。
- **开环 chunk 采集(喂 chunk-WM)**:当前逐帧 `BalancedTerminalDataset` 要求**逐帧** obs_embedding,
  开环(每 8 帧 1 个 embedding)不兼容(§数据契约核实)。**待确认** chunk-WM 是否真按 per-chunk 消费;
  若是,则开环采集是一个**未来采集模式**(需新 dataset/preprocess 对齐 per-chunk obs_embedding),
  8× 省推理。属离线管线改动,非采集器内部。
- **P5 教程**"冷启动并行采集"一节;**严谨吞吐基准**(K=1 vs K=8,大 N 摊薄 spawn/load)。
- **采集器 Hydra 化 + 冷启动 WM 消费对齐**:见
  `docs/history/2026-06-17-coldstart-collector-hydra-and-wm-consumption.md`
  (config 驱动、单一真相源;阻塞于"用哪个 ckpt + 写哪"两个数据决策)。
