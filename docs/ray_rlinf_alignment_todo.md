# Ray 对齐:待实现(DreamerVLA → RLinf 剩余 TODO)

- 日期:2026-06-19
- 用途:把 DreamerVLA 可选 Ray backend 对齐 RLinf **尚未完成**的工作汇总成一份单一 TODO。
- 已完成的部分与设计立场见 **`docs/ray_rlinf_alignment_implemented.md`**;对齐基准是相邻 `RLinf` 仓库。

> **现状**:单机核心对齐已完成(scheduler 骨架、workers、真实 learner 闭环、FSDP/AMP/offload/FSDP2 显存栈、
> collective/bucket/patch/压缩权重同步、模型注册表、手动 config groups)。**剩余收敛为**:
> 真实长跑验证和 benchmark 结论。真实组件/config 绑定、cold-start/online-cotrain 调度重叠、
> per-stage timing、GPU util / 显存指标采集已补到代码与 gated tests。**多节点横向扩展不是
> DreamerVLA 目标,不再作为待实现项推进。**
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

## 下一步执行顺序(用户确认)

> 顺序按"先把真实组件/真实数据接上,再做重叠与性能"排;1(长跑验证)放在 2/3 能真实跑之后。
> 全部单机;每项都要 TDD + 单机可验证的测试,真实 GPU/资产相关的 e2e 默认 `skip`(gated)。

---

### [x] 2. 真实组件接入 Ray cotrain runner(已完成代码 + gated smoke)

**现状**
- `OnlineCotrainRayRunner` 目前只在 tiny test 模型(`dreamervla/workers/actor/_test_models.py` 的
  `TinyWMPOPolicy`/`TinyWMPOWorldModel`/`TinySuccessClassifier`)上跑通,recipe = `configs/experiment/online_cotrain_ray_dreamervla_tiny.yaml`。
- `LearnerWorker._build_components` 用 `_build_from_cfg`(读 `target/_target_/class_path` + `kwargs`)能构建**任意** `nn.Module`,
  并支持按 component 名从 `init_ckpt` `load_state_dict`(`learner_worker.py:115-130`)。tiny models 本就是照真实
  DreamerVLA 协议写的(policy `sample/evaluate`、world_model 的 `encode_latent/observe_next/actor_input/predict_next_chunk` mode、
  classifier `predict_success`),所以接真实组件**主要是写 config + 核对契约**,无需改 learner 代码。

**目标**:一条 `configs/experiment/online_cotrain_ray_oft.yaml` 用真实 VLA / world model / classifier target,
Ray runner 跑通 N 步 smoke。

**步骤**
1. 核对真实模块的构造签名与 forward 契约 ↔ `_dreamervla_{wm,classifier,rl}_update_once`
   (`learner_worker.py:190-259`)与 `inference_worker.forward_batch` 的调用一致(找出 tiny↔real 的协议差并补齐)。
2. 写真实 recipe:`learner.model_cfg.{policy,world_model,classifier}` 指真实 target;`inference.cfg.{encoder,world_model,policy}`
   指真实 target;`env.cfg` 指真实 LIBERO env;`learner.train_cfg`={mode:dreamervla_cotrain, precision, fsdp, algorithm_cfg}。
3. 用 warmup ckpt 走 `init_ckpt`(按 component 名)做组件初始化,而不是随机 init。
4. **TDD**:加 gated e2e(默认 `skip`,需真实 ckpt/env),跑 1–2 步,断言 metrics 含真实 loss 键
   (`wm/loss`/`cls/loss`/`rl/actor_loss`)且有限、非 NaN。

**验收**:真实组件经 Hydra 构建,Ray runner 完成 ≥1 低步数 smoke;不引入新训练拓扑。
**边界/依赖**:单机;依赖真实组件模块 + warmup ckpt(走 config,不硬编码);收敛归 item 1。**成本:中。**

**实现结果(2026-06-19)**:
- 新增 `configs/experiment/online_cotrain_ray_oft.yaml` + `configs/dreamervla/ray_online_cotrain_rynn_action_hidden.yaml`。
  `ray_components.*` 持有模型 target/kwargs,`task`/`env`/`replay` 持有 dataset/task/rollout 信息,实现 model 和 dataset 解耦。
- `OnlineCotrainRayRunner._load_init_ckpt` 支持 runner-format `state_dicts[component]` warmup 初始化;
  inference / learner 分别按 component 名接收 state dict。
- `InferenceWorker` 支持真实 `RynnVLAEncoder.encode(obs)` fallback;runner history 透传 `wm/loss`/`cls/loss`/`rl/actor_loss`。
- Gated e2e: `tests/e2e_tests/test_s5_ray_real_cotrain.py` 默认 skip,设置真实 ckpt/env 后跑低步数 smoke。

**实现细节(spec)**
- 真实 `_target_`(来自单机 `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`,直接 mirror):
  - **policy** `dreamervla.models.actor.RynnVLAActionHiddenActor`(`action_hidden_dim=1024, action_dim=7, time_horizon=5, head_type=legacy, adapter_type=residual_mlp, freeze_output_projection=false, init_action_head_ckpt=${init.vla_ckpt_path}`)
  - **world_model** `dreamervla.models.world_model.dino_wm_chunk.ChunkAwareDinoWMWorldModel`(`chunk_size=5, obs_dim=35840, token_count=35, token_dim=1024, num_hist=3, reward_head_type=binary, freeze_backbone=true`)
  - **classifier** `dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier`(`window=8, granularity=chunk, chunk_size=5, chunk_pool=last, head_type=transformer`)
  - **encoder**(inference 侧)`dreamervla.models.encoder.RynnVLAEncoder`(`model_path=${init.vla_ckpt_path}, action_dim=7, time_horizon=5, pool=mean, freeze_backbone=true`)
- 契约映射(learner 调用点 ↔ 真实模块;tiny↔real 同形,理论上无需改 learner):

  | learner 调用 | 真实模块方法/mode |
  |---|---|
  | `_dreamervla_wm_update_once` → `world_model_pretrain_step` | WM `forward(batch)`(无 mode)→ `{"_loss"/"loss", ...}` |
  | `_dreamervla_rl_update_once` → `dino_wmpo_outcome_step` | WM `mode=predict_next_chunk`(K 步)+ policy `mode=sample`/`evaluate` + classifier `predict_success` |
  | `inference_worker.forward_batch` | encoder + WM `encode_latent`/`observe_next`/`actor_input` + policy `mode=sample` |
- **init_ckpt 键 = component 名** `{"policy","world_model","classifier"}`,值取 warmup ckpt 的 `state_dicts[<name>]`(保存布局见 `online_dreamervla.py:496-521`)。
- **optim**(单机配方):`world_model` adamw lr `2e-6` / `policy` adam lr `5e-7` / `classifier` adamw lr `1e-4`;经 `learner.train_cfg.optimizers.{name}.lr` 传入。
- **gated 测试骨架** `tests/e2e_tests/test_s5_ray_real_cotrain.py`:`@pytest.mark.skipif(not os.environ.get("DVLA_OFT_CKPT"), reason=...)`;`compose(experiment=online_cotrain_ray_oft)` → `runner.run()`;断言 `history` 含 `wm/loss`/`cls/loss`/`rl/actor_loss` 且 `isfinite`。
- ⚠️ 风险点:真实模块的 `hidden` 维度(`token_count×token_dim=35×1024`)与 batch 形状须与 replay 采样吻合;`obs_dim=35840` 与 encoder 输出对齐——这是"核对契约"步骤要逐一确认的地方。

---

### [x] 3. Ray cold-start OFT 按 config 调用(已完成 config/schema + gated smoke)

**现状**
- `cold_start_ray_collect_runner.py` 有 `synthetic` 与 `oft` 两 mode;`_build_oft_components` / `build_oft_worker_plan`
  已能从 `CollectRolloutsRunner._build_collect_cfg()` 装配真实 `OFTRolloutBundle` + LIBERO env + `DumpWorker`,
  recipe = `configs/experiment/collect_rollouts_ray.yaml`(`mode=oft`)。
- 还需把真实 `env.cfg`/`inference.cfg`/`dump.{reward_dir,hidden_dir}` 完整绑进 recipe,并核对落盘 schema。

**目标**:`python -m dreamervla.train experiment=<ray_oft_collect>` 直接产出 reward HDF5 + matching hidden sidecar。

**步骤**
1. 核对 `collect_rollouts_ray.yaml` 的 `collect.*`(`num_images_in_input`、`episodes_per_task`、`envs_per_gpu`、`task_ids`)
   + task recipe(`OpenVLA_Onetraj_ColdStart_LIBERO`)。
2. 确认 dump schema(`preprocess_config.json` + reward HDF5 + obs_embedding sidecar,见 `_make_preprocess_config`)
   与离线消费端 `BalancedTerminalDataset` 一致;sidecar dim 校验通过。
3. 跑一条小 `task_ids` 的真实采集,核对落盘 + sidecar。
4. **TDD**:已有 `tests/e2e_tests/test_s6_ray_coldstart_collect.py`(synthetic + hydra entry);补 gated 真实 OFT e2e(默认 `skip`)。

**验收**:一条命令按 config 输出 reward HDF5 + matching sidecar。
**边界/依赖**:路径全走 config override / task recipe;依赖 `dvla_oft` 环境(transformers fork,见 [[openvla-oft-needs-transformers-fork]])+ LIBERO 资产。**成本:中。**

**实现结果(2026-06-19)**:
- `collect_rollouts_ray.yaml` 的 OFT plan 继续从 task recipe 派生路径、维度、policy mode 和 dump schema。
- 新增 gated e2e `tests/e2e_tests/test_s6_ray_real_oft_collect.py`,真实资产存在时断言 reward HDF5、hidden sidecar、
  `preprocess_config.json` 落盘且 hidden dim 声明一致。

**实现细节(spec)**
- recipe `configs/experiment/collect_rollouts_ray.yaml`(`mode=oft`)已具备:`defaults:[/task: OpenVLA_Onetraj_ColdStart_LIBERO]`;
  `collect.{num_images_in_input=${task.openvla_oft.num_images_in_input}, episodes_per_task, episode_horizon=64, envs_per_gpu, task_ids}`;
  `env.cfg.target=dreamervla.envs.train_env:DreamerVLAOnlineTrainEnv`(`use_from_config=true`);`rollout.{target_episodes,max_steps}`。
- 装配流(`build_oft_worker_plan` → `CollectRolloutsRunner._build_collect_cfg()`):
  - inference.decoder = `dreamervla.workers.inference.oft_rollout:OFTRolloutBundle`(`policy_cfg`/`unnorm_key`/`image_keys`/`history`/`obs_hidden_source`/`expected_*`/`device=cuda`);
  - dump = `{reward_dir=oft.hdf5_reward_dir, hidden_dir=oft.action_hidden_dir, shard_name, preprocess_config, data_attrs={task_suite_name, env_name=libero}}`。
- **dump schema**(`oft_collect_common.make_preprocess_config` → `${hidden_dir}/preprocess_config.json`)须对齐 `BalancedTerminalDataset` 校验:
  `action_head_type`(`oft_l1_regression`/`oft_discrete_token`,按 **detected** policy mode)、`obs_hidden_source`、`prompt_style`、`history`、
  `include_state`、`rotate_images_180`、`time_horizon`、`token_dim`、`action_dim`、`num_images_in_input`、`chunk_size`、`hidden_key="obs_embedding"`、
  `resolution`、`model_path`、`unnorm_key`;`obs_hidden_source==input_token_embedding` 时附 `token_count`/`hidden_dim`。
- **关键 parity**:sidecar 实际 `hidden_dim` == 声明 `hidden_dim`(`BalancedTerminalDataset` 行 238 校验,不等即报错)。
- **gated 测试骨架** `tests/e2e_tests/test_s6_ray_real_oft_collect.py`:skipif 无 OFT ckpt;`compose(experiment=collect_rollouts_ray)` +
  override 小 `collect.task_ids`、`episodes_per_task=1`;断言 reward HDF5 + `preprocess_config.json` 落盘、sidecar `hidden_dim` 一致。

---

### [x] 4. `scheduler/dynamic_scheduler` 深度重叠(已完成 cold-start sync-pipeline)

**现状**
- `scheduler/dynamic_scheduler.py` = `ComponentScheduler`(`ThreadPoolExecutor` 背书,`submit/drain_ready/shutdown` + `ScheduledWork`)。
- `cold_start_ray_collect_runner._run_loop_overlap`(`cold_start_ray_collect_runner.py:341-408`)当前:用 `pending_steps` +
  `ray.wait(num_returns=1)` 重叠 **env-step refs**,但 **inference 是同步阻塞**的(`infer.forward_batch(...).wait()`,行 361),
  每轮只收一个 ready env,`overlap_events` 计数也很粗(行 359–360)。即 RLinf 说的"sync-pipeline"只做了一半。

**目标**:RLinf sync-pipeline parity —— 推理跑 batch *t* 时上一批 env-step 在飞 + 预取下一帧 `current_obs`;仅 dump-size 达标 / `max_steps` / 退出清理时阻塞。

**步骤(TDD)**
1. **RED**:扩 synthetic overlap e2e,断言更强的重叠信号——`time/overlap_events` 随 steps **近线性增长**(而非仅 `>=1`),
   或断言"推理在飞时已有 env-step 在飞"的计数。
2. **GREEN**:重构 `_run_loop_overlap`:
   - 推理改异步(`infer.forward_batch(...).wait_async()` / 经 `ComponentScheduler.submit("infer", ...)`),不再每轮阻塞;
   - 维护双缓冲:`prefetch` 下一批 `current_obs` + 保持上一批 env-step 在飞;
   - `ray.wait(num_returns>1)` 批量收 ready envs(而非逐个);
   - 仅在 `dump.size()>=target` / `steps>=max_steps` / 收尾时阻塞。
   - 可把 infer/env 两类工作的调度下沉进 `ComponentScheduler`(submit + drain_ready),让 runner 只管控制流。
3. 参考 RLinf `EmbodiedRunner.prefetch_train_bootstrap` 的预取形;**不**采用完整 `AsyncEmbodiedRunner` 并发(更大改造)。

**验收**:增强后的 overlap e2e GREEN;`rollout/episodes` 不回退,且有可观测的重叠增益证据(`time/*`)。
**边界/依赖**:只用现有 `ScheduledWork`/`AsyncWork` + `ray.wait`,不引入新并发框架。**成本:中(重叠正确性敏感——注意权重版本/读写竞争,这是唯一需要算法级谨慎的"做"项)。**

**实现结果(2026-06-19)**:
- `_run_loop_overlap` 改成 Ray ObjectRef 事件循环:async inference refs 与 env-step refs 同时在飞,批量 drain ready refs,
  scheduled OFT task 切换仍保持 per-env 串行。
- Synthetic e2e 断言 `time/overlap_events >= rollout/steps - env/num_env_workers`,并检查
  `time/{infer,env_step,dump}_wait_s` 与 ready batch 指标。

**实现细节(spec)**
- **现循环**(简化)`_run_loop_overlap:341-408`:`launch(obs,ids)` 内 `infer.forward_batch(...).wait()`(**阻塞**)→ 派 step;`while` 里 `ray.wait(num_returns=1)` 收**一个** env → 未停则用该 env 的 `next_obs` 再 `launch`。瓶颈:推理阻塞 + 逐个收。
- **目标循环**(双缓冲 + 异步推理):
  1. 保持一个 `infer.forward_batch(...).wait_async()` 句柄在飞(batch *t*),同时上一批 env-step refs(*t-1*)在飞;
  2. `ray.wait(step_refs, num_returns=k)` **批量**收 ready envs;done env 走 `reset_states`;
  3. 用收上来的 `next_obs` 立刻 post 下一次 async 推理(*t+1*),不等当前推理 join 才动 env;
  4. 仅当 `dump.size()>=target` / `steps>=max_steps` 停发,收尾 join 所有在飞 ref。
  - 调度可下沉 `ComponentScheduler`:`submit("infer", forward)` / `submit("env", step)` + `drain_ready()`,runner 只管控制流。
- **重叠正确性(关键)**:cold-start 采集**无 learner 写权重** → **无权重版本竞争**(比在线 cotrain 简单);只需保证**同一 env 的 step 不并发**(per-env 串行)。
- **测试断言(RED)**:`time/overlap_events` 随 `rollout/steps` 近线性(如 `overlap_events >= steps - num_envs`),而非仅 `>=1`;`rollout/episodes` 与同 config 非重叠路径一致。**可在本机 synthetic CPU 路径完整验证。**

**优化空间评估(为什么这是最大、最确定的收益点)**
- 瓶颈是**不同资源相互空等**:LIBERO env step = **CPU** mujoco/osmesa 渲染(`train_env.render_frame`),inference = **GPU** 前向 → 天然可并行;现循环在推理处阻塞,推理时 env 闲、env step 时 GPU 闲。
- **预期增益**:CPU 渲染与 GPU 推理耗时相当时,完整 overlap ≈ **最多 ~2× rollout 吞吐**(把小的藏进大的);一方占大头时受 **Amdahl** 限制于较小一方占比(LIBERO 渲染常是大头,需 benchmark 定)。
- **两种 overlap 要分清**:
  - **infer ↔ env-step**(本项核心)= 真并行(CPU vs GPU),收益确定;
  - **learn ↔ rollout 单 GPU** = 有限——infer/learn 抢**同一张卡**,非真并行,只能藏 CPU env + replay IO + 双缓冲;真正 learn∥infer 并行需**分卡**(多 GPU learner,本仓受限;多节点已是非目标)。
- 顺序:先 cold-start(无 learner 写权重 → 无版本竞争,最干净),再推到在线 rollout 的 infer↔env(= item 6)。

---

### [x] 6. 在线 cotrain rollout 的 infer↔env 深度 overlap(已完成,单机可验证)

**现状**:item 4 已把 **cold-start collect** 的 `_run_loop_overlap` 改成 async ObjectRef 事件循环;**在线 cotrain runner 还没**。
`OnlineCotrainRayRunner._run_loop`(`online_cotrain_ray_runner.py`)里 **learner 已异步**(`pending_learn=learner.update(...)` + `.done()/.wait()`,
即 **learn↔rollout 已重叠**),但 rollout **内部 infer 与 env-step 仍同步**:`infer.forward_batch(...).wait()`(行 ~39)→ `envs...step(...).wait()`(行 ~49),
逐拍 infer→step,**GPU 推理时 env 闲、env-step 时 GPU 闲**。

**目标**:把 item 4 的 sync-pipeline 模式搬到在线 cotrain rollout —— 推理异步在飞 + 上一批 env-step 在飞 + 预取下一帧 obs;
GPU 推理与 CPU env-step 重叠(learner 仍照旧异步,不动)。

**步骤(TDD)**:
1. **RED**:扩 `online_cotrain_ray_dreamervla_tiny` smoke,断言 rollout 内 infer↔env 重叠信号(新增如 `time/rollout_overlap_events` 随 steps 增长),区别于已有的 learner 重叠。
2. **GREEN**:`_run_loop` 复用 cold-start `_run_loop_overlap` 的写法(async `forward_batch` + 批量 `ray.wait` 收 env-step refs + 双缓冲预取),保持 per-env 串行;learner 异步路径不变。
3. 复用已加的 `time/{infer,env_step}_wait_s` instrumentation 验证 GPU 空等下降。

**验收**:tiny synthetic e2e GREEN(rollout 重叠信号增强 + `rollout/episodes`、关键 loss 不回退)。**本机 synthetic CPU 路径可完整验证。**
**边界/依赖**:单机;learn↔rollout 单 GPU 仍只藏 CPU/IO(见 item 4「优化空间评估」),本项只攻 **infer↔env**。**成本:中(重叠正确性敏感:per-env 串行 + 在线路径有权重 sync,注意推理用的是 sync 前/后的权重版本——比 cold-start 多一层需小心)。**

**实现结果(2026-06-19)**:`OnlineCotrainRayRunner._run_loop` 现在真实 Ray WorkerGroup 走
`_run_loop_overlap`:async `InferenceWorker.forward_batch` refs 与 per-env `EnvWorker.step` refs 同时在飞,
`ray.wait` 批量 drain ready refs;同一 env 只在上一步完成后才重新入队,保持 per-env 串行。learner update / weight sync
仍保留原异步路径。新增 `time/rollout_overlap_events`、`time/rollout_strict_overlap_events`、
`time/rollout_{infer,env}_ready_batches` 与 `time/ray_wait_s`;非 Ray fake 单测保留同步回退。
验证:`tests/e2e_tests/test_s5_ray_cotrain_smoke.py` 断言 overlap 事件随 `rollout/steps` 近线性增长。

---

### [ ] 1. 真实 LIBERO/OFT 长跑验证(有空 / 有资源时做)

**现状**:`OnlineCotrainPipelineRunner`(单机)与 Ray tiny 都跑通;真实 OFT/LIBERO 的**收敛/指标**尚未验证
(亦即 `docs/superpowers/TODO/INDEX.md` 里的 "offline-warmup → online-cotrain 真实长跑验证")。

**步骤**:跑真实配置(单机 pipeline 与/或 item 2 的 Ray recipe),记录 `train/ eval/ env/ rollout/ time/` 指标、
收敛曲线、checkpoint/resume 行为;对照 RLinf 并行 eval 基线(libero-goal traj1 ≈ 0.50 `success_once`,见 [[rlinf-parallel-eval-repro]])。

**验收**:一份可复现命令 + run root + 关键指标摘要 + 收敛/失败结论。
**边界/依赖**:验证任务,非新增功能;依赖 item 2/3 能真实跑 + GPU/数据窗口。**成本:高(算力/时间),低代码。**

---

### [ ] 5. 性能优化确认(benchmark 驱动 —— 先量后调)

**已做**:手动性能/显存设施齐(FSDP/FSDP2、AMP、CPU offload、activation checkpointing、collective send/recv、
bucket/patch/compressed weight sync);OFT 推理路径已 `torch.autocast`(`openvla_oft_policy.py:227`),
RynnVLA 的 Chameleon backbone 已用 `attn_implementation="sdpa"`(`rynnvla_encoder.py:146`,PyTorch 融合 SDPA)。
**未做**:无真实 LIBERO/OFT 长跑的吞吐/显存 benchmark;无 kernel 级调优结论。

**kernel 适用性评估(grounding 后,别期待大收益)**

| 手段 | 本仓适用性 |
|---|---|
| `liger_kernel` | ❌ **N/A** —— liger 是 **Qwen 家族**专用;RynnVLA backbone 是 **Chameleon**,不覆盖 |
| FlashAttention-2 | ⚠️ **边际** —— Chameleon 现跑 `sdpa`(已接近 FA);DINO-WM(depth 6,~100 token)/ classifier(4 层)太小,FA 收益可忽略 |
| `torch.autocast(bf16)` | ✅ **已在**(OFT policy + learner AMP) |
| `torch.compile` | 🟡 **候选但不确定**——frozen backbone + WM 可能有 cheap win,但有 graph-break / 动态 shape 风险,需验证 |

**结论**:kernel 大头已被 `sdpa + autocast` 吃掉;真正时间大头很可能是 **frozen Chameleon backbone 推理** + **CPU 渲染**——
前者靠批量 / 缓存 /(分卡),后者靠 **item 4 overlap**,**都不是 kernel 能解决的**。盲做 kernel 优化 ROI 低。

**前置项(应先做)—— instrumentation**:在真实跑里记 per-stage timing(env-step / encode / WM / policy / learner step)
+ GPU 利用率 + `time/` 命名空间指标。**没有这个,item 5 就是盲调。**(instrumentation 逻辑本机可验证,真实数字需 GPU 环境。)

**已补 instrumentation(2026-06-19)**:`InferenceWorker.forward_batch` 记录 `encode_s` / `world_model_s` / `policy_s`;
`OnlineCotrainRayRunner` 汇总到 `time/infer_*_s` 并记录 `time/{infer,env_step,learner,weight_sync}_wait_s`;
cold-start overlap 记录 `time/{infer,env_step,dump,ray}_wait_s`;`dreamervla.utils.resource_metrics`
提供 `nvidia-smi` 聚合 GPU util / memory used / total 和 torch CUDA allocator 当前/峰值显存指标,
并由 online cotrain 与 cold-start runner 合并到 `time/` 命名空间。**未完成**:真实长跑 benchmark
数值和 kernel 开关结论,仍需 item 1/真实资产窗口。

**步骤**:① 加 instrumentation;② 跑真实配置(依赖 item 1/2/3)采各阶段耗时 / 吞吐 / 显存峰值;
③ **只在 benchmark 指出的热点上**决定是否开 `attention_backend=FLASH_ATTN` / `torch.compile`(默认关;`liger` N/A)。
**验收**:一份 benchmark 摘要(各阶段耗时 + GPU util + 显存峰值)+ **有数据支撑**的"是否启用某 kernel"结论。
**边界/依赖**:依赖 item 1/2/3 能真实跑。**成本:中,数据驱动。**

---

## 条件项(默认不做,触发后才做)

### [x] Channel async API —— 已完成
- `channel.py` 已有 key 路由 + weighted batch;已补统一 `AsyncWork` 句柄、`put_no_wait/get_no_wait`
  + batch/weighted-batch no-wait 变体(`channel.py:99-110`)。

### [ ] reward / critic worker(条件)
- **触发**:RL 需要**独立的 reward 服务**(reward model 推理)或**独立的 critic worker**(价值网络)。
- **现状**:当前 outcome reward 在 env/算法内算,critic 不参与动作选择故不进推理路径——**都不需要**。
- **步骤**(触发时):按现有 worker 模式新建 `workers/{reward,critic}/`(`Worker` 子类,reward `forward`→标量 / critic `value`→标量),
  接入 learner/runner;与"内联计算"做 parity(对齐 §P0 的等价测试思路)。
- **验收**:独立 worker 产出的 reward/value 与内联版本逐位一致。

### [ ] hardware 注册表扩展 + 高效 kernel(条件)
- **触发**:上 **NPU / 机器人**等非 CUDA 设备,或要做 **kernel 级加速**。
- **现状**:`scheduler/hardware.py` 只发现 CUDA(`AcceleratorInfo(kind="cuda")`、`discover_local_accelerators`、`count_local_accelerators`),
  且只服务 placement/早校验(**不**自动改 batch/env)。
- **步骤**(触发时):
  - 把 `AcceleratorInfo.kind` 扩成注册表(`cuda`/`npu`/`robot`)+ `HardwareEnumerationPolicy`(对齐 RLinf `rlinf/scheduler/hardware/`),按 kind 插拔发现逻辑;
  - kernel = 加 `use_liger_kernel` / `attention_backend` config 开关(**默认关**,与 RLinf 一致),由 item 5 的 benchmark 决定是否启用。
- **验收**:非 CUDA 设备能被发现/校验;kernel 开关可切换、默认关、不改动 batch/env。

### [ ] Megatron / vLLM / SGLang(条件,默认非目标)
- **触发**:模型**规模/形态**改变 —— 单卡放不下(→ Megatron TP/PP/SP)或策略变**自回归 LLM**(→ vLLM/SGLang 连续批推理)。
- **步骤**(触发时):新建 `hybrid_engines/{megatron,vllm}/`;Megatron 需 `build_transformer_config` /
  `_build_model_parallel_config` 从 cfg 派生 typed config + 早断言(`vocab % tp_size` 等),对齐 RLinf `rlinf/config.py`。
- **默认维持非目标**:DreamerVLA 推理是定长前向产 action chunk(非自回归 token 生成),单卡可放;不改模型形态就不做。

---

## 备注

- 多节点不是目标;单机所有 P0/P1/P3 项都只用 loopback + 共享内存(见 implemented 文档 §4)。
- 所有显存/资源项均为**手动杠杆 + 早校验 + 可观测**,**不做 VRAM 自适应**(见 implemented 文档 §3.1)。
- **性能优化执行计划(先量后调)** —— 用户已确认"能优化最好",据此固化为三步:
  - **O1(现在做,单机可验证)= item 6**:把 cold-start(item 4 已做)的 infer↔env async overlap 搬到**在线 cotrain rollout**。
    这是当前**唯一可现做且确定有收益**的优化(GPU 推理 ∥ CPU 渲染)。learn↔rollout 已异步;learn∥infer 真并行需分卡(非目标)。
  - **O2(需 GPU,gated)= item 1 + item 5**:跑真实 LIBERO/OFT,用已加的 `time/*` instrumentation 采各阶段耗时 + GPU util + 显存峰值 + 吞吐 → 数据。
  - **O3(gated on O2)= item 5 收尾**:**只在 benchmark 指出的热点上**开 kernel 开关(FA2 / `torch.compile`;`liger` 因 Chameleon backbone 而 N/A)。kernel 预期增益不大——大头是 frozen backbone 推理 + CPU 渲染,靠 O1 overlap / 批量解决。
  - 已落地的前置:cold-start overlap(item 4)+ online cotrain rollout overlap(item 6)+ per-stage/resource instrumentation
    (`time/infer_*_s`、`time/{infer,env_step,learner,weight_sync,dump,ray}_wait_s`、GPU util / 显存指标)均在仓。
- 每补一项加配套测试;真实更新步相关回归优先(P0)。
