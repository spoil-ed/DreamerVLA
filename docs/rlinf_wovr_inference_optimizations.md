# RLinf WoVR Inference Optimizations and DreamerVLA Alignment

> 本文档是用户明确要求的跨项目对齐分析，是
> `tests/unit_tests/test_repository_hygiene.py::test_active_sources_do_not_use_removed_rl_route_wording`
> 的一个有意例外（已加入该测试的 skip_paths）。除本文档与 `spec/99_manual_notes.md` 外，
> 活跃源文件仍然禁止出现该外部方案名。

结论先行：DreamerVLA manual Ray cotrain 的 imagine 推理路径与 RLinf WoVR
（`wan_libero_*_grpo_openvlaoft` 路线）的九个推理优化点中，**7 个已对齐或本质等价，
1 个部分对齐（权重同步机制不同但语义等价），1 个未对齐（env bootstrap 与 actor
训练的重叠）**。此外 DreamerVLA 的世界模型 env 是 latent-only stepping，比 RLinf
的像素级视频扩散世界模型推理开销低一个量级——这是 DreamerVLA 领先而非落后的点。

对照基线：DreamerVLA 工作树 2026-07-02（含未提交的 per-rank batch 精修 diff），
RLinf 为与 DreamerVLA 同级的 `RLinf` 工作区。

---

## 1. RLinf WoVR 推理优化点（含文件与类/函数出处）

RLinf 入口 `examples/embodiment/train_embodied_agent.py` 建立四个独立 Ray
WorkerGroup（actor / rollout / env / 可选 reward），由
`rlinf/runners/embodied_runner.py::EmbodiedRunner` 驱动；异步变体为
`async_embodied_runner.py` / `async_ppo_embodied_runner.py`。

| # | 优化点 | RLinf 出处 | 机制摘要 |
|---|--------|-----------|----------|
| 1 | Actor / Rollout 解耦 | `rlinf/workers/actor/fsdp_actor_worker.py::EmbodiedFSDPActor`；`rlinf/workers/rollout/hf/huggingface_worker.py::MultiStepRolloutWorker` | 训练侧持有 FSDP 分片模型 + 优化器；推理侧是独立 HF 权重副本，仅 forward。两组各自 placement，经命名 `Channel` 通信 |
| 2 | rollout 无梯度策略副本 | `MultiStepRolloutWorker.init_worker`（`hf_model.eval()`，独立 precision/model_path）；`.predict`（整体包在 `torch.no_grad()` 内）；可选 `enable_torch_compile` / `capture_cuda_graph` | 推理副本无优化器状态，eval 模式 + no_grad + 独立 dtype |
| 3 | EnvWorker 负责 env step + trajectory assembly | `rlinf/workers/env/env_worker.py::EnvWorker.env_interact_step`、`._run_interact_once`；`rlinf/data/embodied_io_struct.py::EmbodiedRolloutResult.to_trajectory` | EnvWorker 步进向量 env，把 env transition 和 rollout 输出配对成 `ChunkStepResult`，累积为 `[T,B,...]` 的 `Trajectory` 后发给 actor |
| 4 | action chunk 批量推理 | `MultiStepRolloutWorker.predict`（一次 forward 产出 `num_action_chunks` 个动作）；各 env 的 `chunk_step`（如 `rlinf/envs/maniskill/maniskill_env.py:chunk_step`）；`rlinf/envs/action_utils.py::prepare_actions` | env↔rollout 一个 round-trip 对应一个 chunk 而非一个原子步；done 只在 chunk 边界收敛 |
| 5 | pipeline stage / worker group 并行 | `cfg.rollout.pipeline_stage_num`（`EnvWorker.stage_num`、`MultiStepRolloutWorker.num_pipeline_stages`）；`EmbodiedRunner.run` 并发启动 `env.interact` / `rollout.generate` / `actor.recv_rollout_trajectories` | 组间经 Channel 全并发；组内把 env 切成多个 stage 子向量 env，stage i 在仿真时 stage i-1 在做推理 |
| 6 | actor→rollout 权重同步 | `EmbodiedFSDPActor.sync_model_to_rollout` + `MultiStepRolloutWorker.sync_model_from_actor`；`rlinf/hybrid_engines/weight_syncer/bucket_syncer.py::BucketWeightSyncer`（另有 `patch_syncer.py`） | NCCL collective broadcast，按 bucket 流式发送控制峰值显存，payload 内嵌整数 version；频率 `runner.weight_sync_interval` |
| 7 | env bootstrap 重叠 | `EmbodiedRunner.run`（`overlap_env_bootstrap`）；`EnvWorker.prefetch_train_bootstrap` / `_bootstrap_and_send_train` | actor 在 `run_training` 期间，env 组提前 reset 并把下一轮的首个 obs batch 预送到 rollout channel |
| 8 | 世界模型 env stepping | `rlinf/envs/world_model/base_world_env.py::BaseWorldEnv`；`world_model_wan_env.py::chunk_step`（`_infer_next_chunk_frames` 扩散生成帧 + `_infer_next_chunk_rewards`） | **像素级**视频扩散世界模型：动作 chunk → 生成 RGB 帧 → learned reward model 打分。OpenSora 变体内部走 VAE latent，但对 policy 仍物化为帧。**不是** latent-only |
| 9 | trajectory metadata / version 追踪 | `MultiStepRolloutWorker.generate_one_epoch`（`RolloutResult.versions`）；`embodied_io_struct.py::Trajectory.model_weights_id`（`get_model_weights_id` 哈希） | 每条 transition 打上产出它的权重版本；Trajectory 附带版本哈希，供 async 路线判定 off-policy 程度 |

消息粒度：RLinf env↔rollout 走 **per-rank batch**（`EnvWorker.send_env_batch` 按
stage 发送整批 obs dict，`CommMapper` 负责 env:rollout:actor 世界大小不等时的分片），
Ray 控制面只看到粗粒度 WorkerGroup 调用。

---

## 2. DreamerVLA 当前对应位置

manual Ray cotrain 路线：`experiment=openvla_onetraj_libero_cotrain`
（`dreamervla/runners/manual_cotrain_ray_runner.py::ManualCotrainRayRunner`）。
拓扑为 Learner / Actor / Rollout / Env 四组（`_build_groups`，placement 由
`dreamervla/workers/cotrain/placement.py::build_manual_cotrain_placement` 计算：
GPU0 = real_env + rollout0 + learner，GPU>0 = wm_env + rollout + actor）。

| RLinf 优化点 | DreamerVLA 对应 |
|--------------|-----------------|
| 1 Actor/Rollout 解耦 | `dreamervla/workers/actor/embodied_fsdp_actor.py::EmbodiedFSDPActor`（FSDP 训练）与 `dreamervla/workers/rollout/multistep_rollout_worker.py::MultiStepRolloutWorker`（推理副本），独立 WorkerGroup + 三条命名 Channel（env/rollout/actor） |
| 2 no-grad 副本 | `MultiStepRolloutWorker.generate_once/generate_batch/generate_result_batch` 均 `@torch.no_grad()`；`init` 中 `policy.eval()`；输出 detach 到 CPU |
| 3 EnvWorker 组装轨迹 | `dreamervla/workers/env/trajectory_env_worker.py::BaseTrajectoryEnvWorker.interact` 步进 + `_build_trajectory_shard`（per-slot `TrajectoryShard`）→ actor 侧 `messages.py::collate_trajectory_shards` 合成 `[steps,batch,...]` |
| 4 action chunk 批量推理 | rollout 一次 forward `{"mode":"sample","return_chunk":True}` 产出 `[B,K,A]` chunk；WM env `latent_world_model_env.py::chunk_step_batch` 一次 WM forward 步进整个 chunk（`predict_next_chunk`），缺 chunk 模式时回退 `step_batch` |
| 5 组间并行 | `_run_global_step` 并发运行 real env rollout、WM imagine lease 池（`_wait_env_metrics_with_dynamic_wm_leases`）和 actor 轨迹接收（`_start_actor_trajectory_receivers`）。**无组内 pipeline_stage_num 子流水** |
| 6 权重同步 | `EmbodiedFSDPActor.sync_model_to_rollout` → `dreamervla/hybrid_engines/weight_syncer/patch.py::PatchWeightSyncer.push/pull`：命名 Ray object store，支持增量 diff patch，`global_step` 作 version；频率 `manual_cotrain.sync_every` |
| 7 bootstrap 重叠 | **无**：`bootstrap_obs` 在 `interact` 内联执行；episode 终止时 slot 就地 reset（`_step_slot`） |
| 8 WM env stepping | `dreamervla/envs/world_model/latent_world_model_env.py::LatentWorldModelEnv`：**latent-only**（obs 只含 latent/lang_emb/proprio，从不调解码器），reward 来自 classifier 对 latent 打分（`_score_batch`）。比 RLinf 的视频扩散 WM 便宜得多 |
| 9 version 追踪 | rollout 把 policy/wm/classifier 版本写进每条 `RolloutResultMsg.versions` → `TrajectoryShard.versions` → replay sidecars（`_model_version_sidecars`）；checkpoint manifest 记录全部版本。无 RLinf 式 `model_weights_id` 哈希（有 per-component 整数版本，语义等价） |

消息粒度：经近期 batching 改造（HEAD 的 `perf: batch manual cotrain wm env slots`
/ `perf: batch manual cotrain imagine pipeline` + 本次工作树 diff），env↔rollout
主链路已是 **per-rank batch**：`_observation_batch_msg` / `ObservationBatchMsg`
（key=`str(env_rank)`）↔ `MultiStepRolloutWorker._generate_from_rank_batch_key` /
`RolloutResultBatchMsg`。per-slot 路径仅作兼容回退保留。

---

## 3. 差距表

| 优化点 | 状态 | 说明 |
|--------|------|------|
| 1 Actor/Rollout 解耦 | ✅ 已对齐 | 结构一一对应；DreamerVLA 额外多一个 LearnerGroup（WM/classifier 更新），是主线语义要求 |
| 2 no-grad 副本 | ✅ 已对齐 | 差异仅在 RLinf 有可选 torch.compile / CUDA graph 加速（DreamerVLA 未启用） |
| 3 EnvWorker 轨迹组装 | ✅ 已对齐 | DreamerVLA 以 per-slot shard 缓冲 + actor 侧 collate；RLinf 在 env 侧直接拼 `[T,B]`。批量语义等价 |
| 4 action chunk 推理 | ✅ 已对齐 | 两边 env↔rollout round-trip 均为 chunk 粒度；WM env 另有 chunk 级批量 forward（`chunk_step_batch`） |
| 5 组间/组内并行 | 🟡 部分对齐 | 组间并发已对齐（real env ∥ WM lease 池 ∥ actor 接收）；**组内 pipeline_stage_num 子流水未实现** |
| 6 权重同步 | 🟡 机制不同、语义等价 | RLinf：NCCL bucket broadcast；DreamerVLA：Ray object store + 增量 patch。均带整数版本、按间隔触发。单机小权重差距不大；多机/大模型时 NCCL 方案更优 |
| 7 env bootstrap 重叠 | ❌ 未对齐 | RLinf 在 actor 训练期间 prefetch 下一轮 bootstrap；DreamerVLA reset 全部内联 |
| 8 WM env stepping | ✅（DreamerVLA 更优） | DreamerVLA latent-only、无像素解码；RLinf 是像素级视频扩散（每 chunk 跑 5 步扩散推理）。不移植 |
| 9 version 追踪 | ✅ 已对齐 | DreamerVLA 有 per-component 版本并落到 replay sidecar；缺 `model_weights_id` 哈希（可选锦上添花） |
| （消息粒度）per-rank batch | ✅ 已对齐 | 本轮 diff 完成收尾：`_cat_step_batch` 秩归一化、`lang_emb` 等 forward_inputs 的 chunk-batch 整形、中央池化进度 |

不适合直接移植的部分：
- **像素级视频扩散 WM env**（RLinf `world_model_wan_env.py`）：DreamerVLA 的 TSSM
  latent WM + classifier 打分是刻意的低成本设计，移植视频扩散违背主线语义。
- **WM env 的 reward/success**：两边都来自 learned model（RLinf 是 reward model，
  DreamerVLA 是 classifier），**都不是真实 LIBERO success rate**。DreamerVLA 的
  `eval/*` 指标取自 RealEnvWorker 的真实 LIBERO episode 结果，保持这一区分。
- **组内 pipeline_stage_num**：DreamerVLA 单 EnvWorker 内 slot 数少（8），仿真与
  推理延迟不在同一量级（real env 为 LIBERO CPU 仿真、WM env 为轻量 latent forward），
  子流水收益有限、复杂度高，不作为最小改动方向。

可最小改动对齐的方向（按性价比排序）：
1. **env bootstrap overlap**（差距 7）：在 actor `run_training` 期间预取下一
   global step 的 reset/bootstrap obs，对应 RLinf `prefetch_train_bootstrap`。
2. `model_weights_id` 式轨迹级权重哈希（差距 9 的收尾，很小）。
3. rollout 推理副本可选 `torch.compile`（差距 2 的收尾，需 GPU 验证收益）。

---

## 4. 本轮已完成的改动与测试结果

本轮（2026-07-02）在既有未提交 batching diff 之上做的最小修正与验证：

1. `tests/unit_tests/test_cotrain_messages.py::test_collate_trajectory_shards_normalizes_trailing_singleton_rank`
   —— 修复对 float32 张量做精确相等断言导致的假失败（改用 `torch.allclose`）。
2. `tests/unit_tests/test_repository_hygiene.py` —— 将本报告加入
   `test_active_sources_do_not_use_removed_rl_route_wording` 的 `skip_paths`
   （用户明确要求本文档使用该外部方案名）。
3. 验证既有 diff 已满足三项运行时要求：
   - **每 global_step 一次 eval SR**：`manual_cotrain.eval_interval_global_steps: 1`
     （`configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`），
     `_real_env_success_rate_metrics` 把真实 LIBERO episode 成功率镜像到 `eval/*`；
   - **per-rank 批量处理**：`ObservationBatchMsg`/`RolloutResultBatchMsg` 主链路 +
     `_cat_step_batch` 秩归一化 + `_one_forward_input_chunk_batch` 整形；
   - **中央统一进度**：worker 只写 JSON 进度文件，runner 端
     `central_progress_snapshot`/`report_central_progress` 聚合 real_env + wm_pool
     后渲染单一进度行，无 per-worker 子进度条。

同日第二轮（imagine 路径提速，按计划
`docs/superpowers/plans/2026-07-02-manual-cotrain-imagine-speedup.md` TDD 执行）：

4. **imagine 批量对齐**：`manual_cotrain.wm_envs_per_worker` 8 -> 16
   （`configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`；测试
   `test_manual_cotrain_oft_wm_env_num_envs_tracks_wm_envs_per_worker`）。
   实测（下文 run B）表明单纯扩大批量吞吐不变——每次迭代耗时随 payload
   线性增长，证实瓶颈是消息序列化/CPU 而非前向延迟。
5. **消除冗余 hidden 回传**：当 hidden 由 env 侧 obs 提供（imagine 路径恒真）
   时，rollout 不再把 `[B, 56*4096]` 的 hidden 回传（`generate_result_batch`），
   env 在 `_get_rollout_result_batch` 里用自己持有的 obs hidden 原值补回
   shard（值逐位相同）。砍掉 rollout→env 方向最大的张量序列化。
   测试：`test_generate_batch_uses_batched_obs_hidden_payload`（契约翻转）、
   `test_get_rollout_result_batch_injects_hidden_from_slot_obs`。
6. **WM chunk 回退可见化**：`LatentWorldModelEnv._chunk_step_batch_fallback`
   首次触发时打一条 warning（`test_chunk_step_batch_fallback_warns_once`），
   防止静默丢失 chunk 批量。
7. 其他既有失败修复：教程精简后的断言更新
   （`test_openvla_onetraj_tutorial_prefers_role_based_wm_route_examples`）；
   活跃源里的外部方案名措辞清理。

按用户约束：优化聚焦 WM env imagine 部分；imagine step 语义不追求与外部
方案形式对齐（原计划的 real-env bootstrap prefetch 已放弃）。

测试（`dreamervla` conda env）：

```
pytest tests/unit_tests/test_cotrain_messages.py \
       tests/unit_tests/test_latent_world_model_env.py \
       tests/unit_tests/test_trajectory_env_worker.py \
       tests/unit_tests/test_multistep_rollout_worker.py \
       tests/unit_tests/test_manual_cotrain_ray_runner.py -q
# 116 passed（第二轮改动后）
pytest tests/unit_tests -q   # 全量见第 5 节后记
```

## 5. 端到端耗时对照（进行中）

2026-07-02 起在同机启动两侧完整训练做耗时对照（按用户指定使用 GPU 2,3,6,7）：

- RLinf WoVR：`wan_libero_goal_grpo_openvlaoft`（GPU 2,3，2 卡，
  容器 `rlinf-wovr-local:wan`，`--security-opt seccomp=unconfined`，
  日志 `RLinf/logs/20260702-224023-wovr-goal-train-full/`）。
- DreamerVLA：`openvla_onetraj_libero_cotrain` 全量默认参数
  （GPU 6,7，2 卡，init = `ray6_wovr_full_20260628_020057` warmup ckpt，
  输出 `data/outputs/manual_g67_full_20260702_222900/`）。

两侧同为 2 卡。启动期 debug 记录：

1. 容器默认 seccomp 会拦截 CUDA 共享存储用的 `pidfd_getfd`
   （smoke 可能报 `RuntimeError: pidfd_getfd: Operation not permitted`），
   必须 `--security-opt seccomp=unconfined`。
2. 首次启动失败：`ModuleNotFoundError: No module named
   'diffsynth.models.reward_model'` —— wan world-model env 依赖带
   reward_model 的修改版 DiffSynth（`wovr_assets/src/diffsynth-studio-main`），
   需把该目录加入 PYTHONPATH 后重启。
3. 同机另有他人 `lm_eval` 任务占用 GPU 4,5 并在 GPU 6,7 上各驻留
   ~18-20GB（22:35 起），对 DreamerVLA 侧计时有干扰，比较结论需注明。

实测（2026-07-02 深夜，均为 2 卡；imagined env-steps/s 按
chunk/s × num_action_chunks(8) 折算）：

| 侧 | 配置 | 度量 | 吞吐 |
|----|------|------|------|
| 对照方案 (GPU 2,3) | 32 envs（64 在 2 卡 OOM，降半）| 87 s / rollout epoch（32×256=8192 env-steps）| **~94 imagined env-steps/s**（视频扩散 WM + reward model）|
| DreamerVLA run A (GPU 6,7) | wm 8 slots，echo 回传在 | wm_pool 13824 chunks / 22min ≈ 10.5 chunk/s | ~84 imagined env-steps/s（latent WM）|
| DreamerVLA run B (GPU 6,7) | wm 16 slots，echo 回传在 | wm 4096 chunks / 7.5min ≈ 9.1 chunk/s | ~73 imagined env-steps/s |
| DreamerVLA run D (GPU 6,7) | wm 16 slots + 无 hidden 回传 | Δ3072 chunks / 248 s ≈ 12.4 chunk/s | **~99 imagined env-steps/s（比 run B +36%）** |

（run C = run D 同配置的首次启动，因回传消除首版对 bf16 obs 张量调用
`np.asarray` 崩溃（`Got unsupported ScalarType BFloat16`）而重启；已修复并加
回归测试 `test_get_rollout_result_batch_injects_bf16_tensor_hidden`。）

关键读数：
- run A→B 批量翻倍但吞吐不升——**每次批量迭代耗时随 payload 线性增长**，
  imagine 是 env-worker 侧 CPU/序列化瓶颈（WMEnvWorker 单核 ~58% CPU、
  GPU 利用率个位数），不是策略前向瓶颈。这是 run C（回传消除）的动机。
- 对照方案用重一个数量级的世界模型（5B 视频扩散，5 步去噪/chunk）仍达到
  ~94 env-steps/s：其 env↔rollout 单向 payload 只有图像帧（uint8）+ 动作，
  且 EnvWorker 内部 GPU 常驻批量，序列化开销占比低。
- 干扰因素：同机他人 `lm_eval` 任务驻留 GPU 6,7（各 ~19-21GB，间歇性
  90%+ util），DreamerVLA 侧数字系统性偏低；对照侧 GPU 2,3 独占。
- 事故记录：22:57 同账号他人误 `kill` 了 run A 的 Ray worker（SIGTERM，
  journalctl 可查 `sudo kill <rollout pids>` 记录），非代码问题；已重启。

补充（2026-07-03 00:20，两侧训练已按用户指示中止，日志保留）：
- 对照方案完整一步：rollout 23.3min + actor update ≥23min（被停时未完成），
  即 2 卡 32 envs 下 **单 global step ≥ 45-50min**。
- DreamerVLA 6 卡全量估算：WM imagine ≈ 18min（5 worker 分摊）∥ real env
  ≈ 90min（瓶颈）+ 更新阶段数分钟 → **~1.5-2h/步**；据此按用户指示将
  `manual_cotrain.real_rollout_epoch` 4 → 1（real 轨迹 32 → 8/步，
  8+1024=1032 可被 group_size=8 整除），real env 阶段降至 ~23min，
  6 卡预期 **~25-30min/步**，与对照方案同量级。
  测试：`test_manual_cotrain_real_rollout_budget_is_one_epoch`。

结论（截至 2026-07-03 00:00）：
- 消除 hidden 回传后，DreamerVLA imagine 吞吐 ~99 imagined env-steps/s，
  与对照方案 rollout 阶段的 ~94 env-steps/s 相当且略优——而且 DreamerVLA 侧
  还承受着 lm_eval 的 GPU 争抢；瓶颈定位（payload 线性的 env-worker CPU）
  与优化方向得到实测闭环验证。
- 对照方案单个 global epoch = 16 rollout epoch × 87.3s ≈ 23.3min（+ actor
  update，未计完）；DreamerVLA 单 global step 消耗 4 倍 imagined env-steps
  （1024 traj × 512 步 vs 512 traj × 256 步），直接比较 wall time/step 无意义，
  应比较 env-steps/s（上表）。
- 全量单测：`tests/unit_tests` **1328 passed, 7 skipped, 0 failed**
  （2026-07-03 00:00，dreamervla env）。

## 6. 风险与后续工作

- **bootstrap overlap 未实现**：每个 global step 头部有一段串行 reset 时间，
  规模大时可见；是下一轮最小改动候选。
- **双路径并存**：per-slot 兼容路径（`_generate_from_key`、
  `apply_rollout_result` 单 slot 分支）仍在。功能上无害，但增加维护面；
  等 per-rank 路径 GPU 全量验证后可清理。
- **WM chunk 模式是 best-effort**：`_looks_like_missing_chunk_mode` 的静默回退
  可能掩盖配置错误（想要 chunk 批量却拿到 per-step）；可考虑在 init 时打一条
  rank-0 提示。
- **权重同步走 Ray object store**：单机可行；若走多机需评估切换到
  NCCL bucket broadcast（RLinf `BucketWeightSyncer` 形态）。
- **2 卡全量 cotrain 未经验证**：actor 单 rank 承担全部 PPO 批量，存在 OOM
  风险（参照 MEM-RL-01 micro-batch 经验），训练监控中如出现将调
  `algorithm.*`/micro-batch 参数。
