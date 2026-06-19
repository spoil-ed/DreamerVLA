# Ray 后端对齐报告:DreamerVLA 直接对齐 RLinf

- 日期:2026-06-19(2026-06-19 重定向:由"差异对比"改为"**以直接对齐 RLinf 为目标**的现状 + 补齐方案")
- 目标:让 DreamerVLA 的可选 Ray backend **直接对齐 `/mnt/data/spoil/workspace/RLinf` 的工程架构与能力**——
  不再把 RLinf 的分布式 / 显存优化栈当作"刻意省略",而是逐项列为**待对齐的补齐目标**并给出方案与优先级。
- 基准:`RLinf`(`main`)的 `rlinf/scheduler/`、`rlinf/workers/`、`rlinf/hybrid_engines/`、
  `rlinf/runners/`、`ray_utils/`、`pyproject.toml`。
- 现状:`DreamerVLA`(`main`)的 `dreamervla/{scheduler,workers,hybrid_engines,runners}/`、
  `configs/experiment/*ray*`、`tests/{unit_tests,e2e_tests}`。
- 设计依据(本仓):`docs/superpowers/specs/2026-06-17-ray-online-cotrain-backend-design.md` 及 S1–S6 spec(gitignored)。

> **一句话**:DreamerVLA 已经按 RLinf 命名/分层落好了**骨架**(`scheduler`/`worker`/`worker_group`/`channel`/
> `placement`、`workers/{env,inference,actor,replay}`、`hybrid_engines/weight_syncer`、RLinf 风格 runner)。
> **要直接对齐 RLinf,还差三大块**:① 分布式通信栈(collective/NCCL、manager 层、多节点 node 管理、hardware 探测、
> dynamic scheduler);② **显存优化栈(FSDP/Megatron/AMP/激活重计算/CPU offload/liger/flash)**;③ 真实训练闭环
> (learner 现为 synthetic PPO,未接真实更新步、未做训练 parity)。本报告逐项给补齐方案。
>
> ⚠️ **对齐边界提醒**:RLinf **本身也不做"自动适应显存"**——没有 auto-batch-size、没有 OOM-retry、不用
> DeepSpeed/ColossalAI/Lightning。它提供的是**手动 config 驱动**的显存优化构件。所以"对齐 RLinf 的显存能力"
> 指的是**移植这套手动栈**,**不是**去造一个 VRAM 自适应器(见 §4)。

---

## 0. 对齐目标与现状速览

| 维度 | RLinf(对齐基准) | DreamerVLA 现状 | 直接对齐动作 | 优先级 |
|---|---|---|---|---|
| Ray 依赖定位 | **核心必装** `ray[default]>=2.47.0` | **opt-in extra** `[ray]`,import 隔离 | 决策点:是否升核心(§1,默认保留 opt-in) | — |
| `Cluster` | 多节点单例 + NodeProbe + manager 启动 | 单机幂等单例(`num_nodes=1`) | 加多节点 init / 节点探测 / manager 拉起 | P2 |
| `Worker`/`WorkerGroup` | actor 基类 + 组广播 + send/recv 路由 | 同形(广播/`execute_on`/`ray.get`) | 补点对点 `send/recv` + NodeAffinity 调度 | P2 |
| `Placement` | Packed/Node/**Flexible** + 范围语法 + 跨节点 bundle | Packed/Node(单机,ray-free 可单测) | 补 Flexible + 范围语法 + 多节点 bundle | P2 |
| `Channel` | actor FIFO + **key 路由 + 加权批** + AsyncWork | actor FIFO + `get_batch(n)` | 补 key 路由 / 加权批 / 统一异步句柄 | P3 |
| **Collective / NCCL** | `collective/`(NCCL/Gloo,多通道,broadcast/send/recv) | ❌ 无(object-store 代替) | 新建 `scheduler/collective/`(NCCL/Gloo) | P1 |
| **Manager 层** | Worker/Collective/Node/Lock manager(named actor) | ❌ 无 | 新建 `scheduler/manager/`(多节点协调/锁) | P2 |
| **多节点 / Node 管理** | NodeProbe + 节点组 + 跨机 env/解释器 | ❌ localhost only | NodeProbe + 节点组 config + bootstrap | P2 |
| **Hardware 探测** | GPU/NPU/机器人注册表 | ❌ `num_gpus` 由策略写死 | 新建 `scheduler/hardware/`(枚举/排序) | P3 |
| **Dynamic scheduler** | 组件级预取/流水线编排 | ❌ runner 内手动 `ObjectRef` 重叠 | 新建 `scheduler/dynamic_scheduler/` | P3 |
| **显存:分片(ZeRO 等价)** | FSDP(FSDP1+2)/ Megatron(TP/PP/SP) | ❌ learner 单卡裸 Adam | 新建 `hybrid_engines/fsdp/`(FSDPModelManager) | **P0/P1** |
| **显存:CPU offload** | FSDP `CPUOffloadPolicy` + 优化器 offload | ❌ | 随 FSDP 一并接 | P1 |
| **显存:混合精度** | AMP autocast+GradScaler / FSDP MixedPrecision | ❌ | learner 接 autocast + dtype 策略 | P0/P1 |
| **显存:激活重计算** | `gradient_checkpointing_enable` / Megatron recompute | ❌ | 模型支持 + config 开关 | P1 |
| **显存:高效 kernel** | liger_kernel / FlashAttention | ❌ | 可选接(默认关,RLinf 同) | P3 |
| 权重同步 | NCCL 默认 + bucket/patch/压缩 | object-store(CPU sd,单调版本) | 加 NCCL `WeightSyncer` 第二实现 | P1 |
| Workers 角色 | actor/env/rollout/**reward/critic** | env/inference/replay/actor(单卡) | 补 reward/critic worker(若 RL 需要) | P3 |
| Runner | `EmbodiedRunner`(infer→step→learn) | `OnlineCotrainRayRunner` + `ColdStartRayCollectRunner` | 接真实更新步 + S5 parity | **P0** |
| ray_utils / 启动 | `start_ray.sh`/`check_ray.sh`(head/worker 跨机) | 无(`Cluster` 内 `ray.init`) | 多节点时补启动脚本 | P2 |
| cluster 配置 | `ClusterConfig`/`NodeGroupConfig` | experiment YAML 扁平 | 引入 `scheduler/` config 组 + 早校验 | P2 |
| vLLM / SGLang / Megatron | ✅ `hybrid_engines/{vllm,megatron}` | ❌ | **条件对齐**:仅当策略变自回归 LLM / 上大模型 | P3(条件) |

> 优先级口径:**P0 = 阻塞一切的训练正确性**;**P1 = 单机即可见收益的核心对齐(显存栈 + NCCL)**;
> **P2 = 多节点横向扩展**;**P3 = 重型/条件项**。

---

## 1. 依赖与定位(对齐的第一个决策点)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 声明 | core `dependencies`:`ray[default]>=2.47.0`(scheduler 的基石) | `[project.optional-dependencies] ray`;基础依赖仅 `hydra-core`/`omegaconf` |
| 隔离 | 无(全框架 ray-first) | `scheduler/__init__.py` 禁止单机 torchrun 路径 import 本包 |

**直接对齐 RLinf 会要求**:把 ray 提升为**核心必装依赖**(与 RLinf 一致),去掉 import 隔离。

> **建议(需用户拍板)**:**默认保留 opt-in**。RLinf 是 ray-first 分布式框架,ray 当核心理所当然;
> DreamerVLA 是单机训练框架 + 可选 ray 扩展,把 ray 升核心会让纯单机用户被迫装 ray 全家桶、失去"单机零 ray 开销"。
> 工程组织已与 RLinf 对齐,**依赖定位是定位差异、不是能力差距**。仅当多节点成为主线时再升核心。

---

## 2. Scheduler 原语对齐

> 骨架已对齐(`Worker`/`WorkerGroup`/`Channel`/`Placement` 形态与公共 API 几乎一一对应,`Placement` 抽成
> ray-free 模块反而比 RLinf 更干净)。下面只列**补齐项**。

### 2.1 Cluster(P2)
- RLinf:`rlinf/scheduler/cluster/cluster.py` —— `ray.init(address="auto")`、`NodeProbe` 远程探测每节点、
  在 rank-0 拉起 Worker/Collective/Node/Lock manager、SIGUSR1 优雅退出、`get_alive_nodes/get_node_info/get_node_ip`。
- DVLA:`dreamervla/scheduler/cluster.py` —— 单机幂等 `ray.init`、`num_nodes` 硬编码 1、版本断言、`find_free_port`。
- **对齐动作**:`num_nodes` 改为探测;加 `address="auto"` 连接已起集群;新增 NodeProbe(见 §3 多节点)+ manager 启动钩子。

### 2.2 Worker / WorkerGroup(P2,基本已齐)
- RLinf 多了 `send/recv/send_tensor` 点对点(经 WorkerManager 路由)+ `NodeAffinitySchedulingStrategy` 硬绑节点。
- **对齐动作**:在 `worker.py` 加点对点 `send/recv`(多节点时用,单机仍走 `Channel`);`worker_group.launch` 支持
  `NodeAffinitySchedulingStrategy`(依赖 §3 节点信息)。

### 2.3 Placement(P2)
- **对齐动作**:`placement.py` 补 `FlexiblePlacementStrategy` + 范围语法(`"0-3,5,7-9"`)+ 跨节点 bundle(`(node_rank, gpu_bundle)`)。

### 2.4 Channel(P3)
- **对齐动作**:`channel.py` 补 key 路由、加权批(`get_batch(weight=...)`)、统一 `AsyncWork` 句柄;非阻塞 `put_no_wait/get_no_wait`。

### 2.5 RLinf 有、DVLA 无的 scheduler 子系统(补齐方案)

| 子系统 | RLinf 提供 | 直接对齐动作 | 优先级 |
|---|---|---|---|
| `collective/`(NCCL/Gloo) | `Collective`/`CollectiveGroup`/`MultiChannelProcessGroup`:`broadcast/send/recv`、多通道、ring broadcast、CPU↔object-store / GPU↔NCCL | 新建 `dreamervla/scheduler/collective/`;先支撑 NCCL 权重同步(§5),再支撑多卡 learner | **P1** |
| `manager/` | Worker/Collective/Node/DeviceLock/PortLock manager(named actor 全局元数据/路由/锁) | 新建 `dreamervla/scheduler/manager/`;单机可空实现,多节点接 send/recv 路由 + 设备锁 | P2 |
| `cluster/node.py` 多节点 | NodeProbe 远程探测、节点组、跨机 env_vars/解释器、Nsight 包裹 | 接入 §2.1 Cluster;新增节点组 config(§7) | P2 |
| `hardware/` | GPU(N/AMD/Intel)/Ascend NPU/机器人注册表,自动枚举排序 | 新建 `dreamervla/scheduler/hardware/`;最小先做 NVIDIA 枚举,替掉 `num_gpus` 写死 | P3 |
| `dynamic_scheduler/` | `ComponentManager` 组件级预取/流水线编排 | 新建 `dreamervla/scheduler/dynamic_scheduler/`;把 runner 内手写的 `ObjectRef` 重叠抽成调度器 | P3 |

---

## 3. Workers 对齐

| RLinf `rlinf/workers/` | DreamerVLA 现状 | 直接对齐动作 |
|---|---|---|
| `env/`(`EnvWorker`/`AsyncEnvWorker`) | `env/env_worker.py`(每 env 一 actor,done 自动 reset) | **已齐**;可选补 `AsyncEnvWorker` |
| `rollout/`(`MultiStepRolloutWorker`/`SGLangWorker`) | `inference/{inference_worker,rollout_inference_worker}.py`(encoder+WM+policy 同 actor) | **已齐**(刻意不拆推理链);`SGLangWorker` 仅自回归 LLM 才需(条件) |
| `data`(replay) | `replay/replay_worker.py`(包 `OnlineReplay`) | **已齐** |
| `actor/`(FSDP 训练:`learn/get_weight/set_weight`) | `actor/learner_worker.py`(**单卡裸 Adam,synthetic PPO**) | **核心差距**:① 接真实更新步(§6);② 接 FSDP 显存栈(§4) |
| `reward/`(`EmbodiedRewardWorker`) | ❌(outcome reward 在 env/算法内) | 若需独立 reward 服务再补(当前 outcome 模式不需要) |
| `critic/`(FSDP 价值网络) | ❌(critic 不参与动作选择,推理不传) | 若 RL 需独立 critic worker 再补 |

---

## 4. 显存优化栈对齐(一等维度,P0/P1)

> 这是"直接对齐 RLinf"在**单机即可见收益**的核心一块。RLinf 的显存优化是**手动 config 驱动**的整套构件;
> DVLA 的 ray `LearnerWorker` 目前**一样都没有**(单卡、裸 Adam、synthetic PPO)。
> **再次强调对齐边界**:RLinf 没有 auto-batch-size / OOM-retry / DeepSpeed/Lightning,所以这里的"对齐"
> = **移植手动栈**,batch size 仍由 config 手填。

| 能力 | RLinf 实现(基准) | 直接对齐动作(DVLA) | 优先级 |
|---|---|---|---|
| 参数/梯度/优化器分片(ZeRO 等价) | `hybrid_engines/fsdp/`(`fsdp.py`=FSDP1、`fsdp2.py`=FSDP2、`fsdp_model_manager.py`、`strategy/`) | 新建 `dreamervla/hybrid_engines/fsdp/`,提供 `FSDPModelManager` 包 learner 的 policy/WM;`worker_group` 已能多卡分配,补 DDP/FSDP 初始化 | **P1** |
| CPU offload | FSDP `CPUOffloadPolicy` + `optimizer_cpu_offload`/`offload_fraction` | 随 FSDP 一并接;config `learner.fsdp.cpu_offload` | P1 |
| 混合精度(AMP) | `amp_autocast`(autocast+`GradScaler`)/ FSDP `MixedPrecision`(param/reduce/buffer dtype);**默认关** | learner update 包 `torch.autocast` + `GradScaler`;config `learner.precision=bf16` | **P0/P1** |
| 激活重计算 | `module.gradient_checkpointing_enable()`(FSDP 路径)/ Megatron `recompute_granularity/method/num_layers` | 给 WM/policy 模块加 `gradient_checkpointing_enable`;config 开关 | P1 |
| 高效 kernel | `use_liger_kernel`(默认 False)/ FlashAttention(`attention_backend="FLASH_ATTN"`) | 可选接,默认关(与 RLinf 一致) | P3 |
| 张量/流水并行 | `hybrid_engines/megatron/`(TP/PP/SP + microbatch + sequence packing + 优化器 offload) | **条件对齐**:仅当单卡放不下、需大模型并行时引入 | P3(条件) |

**落点建议**:在 `dreamervla/hybrid_engines/` 下镜像 RLinf 的 `fsdp/` 子树(`FSDPModelManager` + `strategy/{base,fsdp,fsdp2,checkpoint}.py`),
`LearnerWorker.init()` 用它包装 policy/WM;`learner.train_cfg` 暴露 `fsdp/precision/cpu_offload/activation_checkpointing` 开关并入 `validate_cfg` 早校验。

---

## 5. 权重同步对齐(P1)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 接口 | `WeightSyncer`(ABC)+ `BucketWeightSyncer`/`PatchSyncer`/`Compressor` | `WeightSyncer`(ABC,`push/pull`) |
| 传输 | **NCCL collective 默认**(ring broadcast)+ object-store(CPU/异构回落) | **仅** object-store(`_WeightStore` actor,单调版本,CPU `state_dict`) |

**直接对齐动作**:沿现有 `WeightSyncer` ABC **加 NCCL 第二实现**(依赖 §2.5 `collective/`):learner→inference 走 broadcast,
object-store 作 CPU/异构回落;可选 bucket 分桶。接口已预留,**追加实现即可、无需重写**。

---

## 6. Runner / 训练闭环对齐(P0,最高优先)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 主循环 | `EmbodiedRunner`:init Cluster → launch group → Channel/WeightSyncer 协调 → 收指标 | `OnlineCotrainRayRunner`:**形态已对齐**(infer→step→learn 重叠、`time/overlap_events`) |
| 学习步 | 真实 PPO/SAC/NFT 更新(FSDP) | **`LearnerWorker.update` = synthetic PPO(对 mean action 做 MSE)** |

**直接对齐动作(阻塞项)**:
1. 把 `LearnerWorker.update` 从 synthetic PPO 换成**真实 DreamerVLA RL / WM / classifier 更新步**(复用单机 `online_dreamervla.py` 的更新逻辑)。
2. 补 **S5 训练等价 parity**:同 config/seed 下,ray backend 的 loss/指标对齐单机 `OnlineCotrainPipelineRunner` 到容差内。
3. 产出**可观测重叠证据**(`time/` 命名空间下 infer/learner 并发占比),否则 ray 仅是复杂化(设计 §12)。

> 此项不补,§4/§5 的显存与 NCCL 对齐都是在错误的训练上加速,**必须先行**。

---

## 7. ray_utils / 启动 / config 对齐(P2)

| | RLinf(基准) | 直接对齐动作 |
|---|---|---|
| 启动脚本 | `ray_utils/start_ray.sh`(rank0 `ray start --head`,worker 轮询 head IP 后 `--address`)、`check_ray.sh`(轮询 `ray status` 等 GPU) | 多节点阶段补 `scripts/start_ray.sh`/`check_ray.sh`;单机仍由 `Cluster` 内 `ray.init` 管 |
| cluster 配置 | `ClusterConfig`/`NodeGroupConfig`:`num_nodes`、`component_placement`、`node_groups`(跨机硬件/env/解释器)、`nsight` | 引入 `configs/scheduler/` 组(对齐 RLinf 的 cluster 配置);`validate_cfg` 校验节点组/placement/GPU 数 |

---

## 8. 测试对齐(随各项推进)

现状:7 单测契约(scheduler/placement/worker 公共 API,ray-free)+ 6 e2e(真 ray smoke/parity,`test_s1..s6`)。
**对齐动作**:每补一项加配套测试——FSDP/AMP 显存栈接入后补显存峰值回归;NCCL `WeightSyncer` 接入后补 object-store/NCCL 双实现 parity;
真实更新步接入后把 `test_s5_*` 从 synthetic 升级为**对单机基线的训练等价 parity**;多节点接入后补跨节点 smoke。

---

## 9. 命名对齐表(RLinf → DreamerVLA;"待建"= 补齐目标落点)

| RLinf 路径 | DreamerVLA 路径 | 状态 |
|---|---|---|
| `rlinf/scheduler/cluster/cluster.py` | `dreamervla/scheduler/cluster.py` | 已齐(单机)→ 补多节点 |
| `rlinf/scheduler/worker/{worker,worker_group}.py` | `dreamervla/scheduler/{worker,worker_group}.py` | 已齐 → 补 send/recv |
| `PlacementStrategy`(worker 子系统内) | `dreamervla/scheduler/placement.py`(独立) | 已齐 → 补 Flexible/多节点 |
| `rlinf/scheduler/channel/channel.py` | `dreamervla/scheduler/channel.py` | 已齐 → 补 key 路由/加权批 |
| `rlinf/scheduler/collective/` | `dreamervla/scheduler/collective/` | **待建(P1)** |
| `rlinf/scheduler/manager/` | `dreamervla/scheduler/manager/` | **待建(P2)** |
| `rlinf/scheduler/{dynamic_scheduler,hardware}/` | `dreamervla/scheduler/{dynamic_scheduler,hardware}/` | **待建(P3)** |
| `rlinf/workers/env/` | `dreamervla/workers/env/env_worker.py` | 已齐 |
| `rlinf/workers/rollout/` | `dreamervla/workers/inference/{inference_worker,rollout_inference_worker}.py` | 已齐 |
| `rlinf/workers/actor/`(FSDP) | `dreamervla/workers/actor/learner_worker.py` | 骨架齐 → 接真实更新步 + FSDP |
| `rlinf/workers/{reward,critic}/` | `dreamervla/workers/{reward,critic}/` | 待建(条件) |
| `rlinf/hybrid_engines/weight_syncer/`(NCCL) | `dreamervla/hybrid_engines/weight_syncer/`(object-store) | 已齐 → 补 NCCL 实现 |
| `rlinf/hybrid_engines/fsdp/` | `dreamervla/hybrid_engines/fsdp/` | **待建(P0/P1)** |
| `rlinf/hybrid_engines/{megatron,vllm}/` | — | 条件对齐(P3) |
| `rlinf/runners/embodied_runner.py` | `dreamervla/runners/online_cotrain_ray_runner.py` | 形态齐 → 接真实更新步 |
| `ray_utils/{start_ray,check_ray}.sh` | `scripts/{start_ray,check_ray}.sh` | 待建(P2) |

---

## 10. 对齐路线图(按优先级)

**P0 —— 训练正确性(阻塞一切)**
- [ ] `LearnerWorker.update`:synthetic PPO → 真实 DreamerVLA RL/WM/classifier 更新步(复用 `online_dreamervla.py`)。
- [ ] S5 训练等价 parity:对单机 `OnlineCotrainPipelineRunner`,同 seed loss/指标对齐到容差。
- [ ] 可观测重叠证据(`time/` 并发占比)。
- [ ] learner 接混合精度(AMP autocast + GradScaler / dtype 策略)——与真实步一起落地最省返工。

**P1 —— 单机可见收益的核心对齐**
- [ ] `hybrid_engines/fsdp/`:镜像 RLinf `FSDPModelManager`,learner 接 FSDP 分片 + CPU offload + 激活重计算。
- [ ] `scheduler/collective/`:NCCL/Gloo collective group。
- [ ] `WeightSyncer` NCCL 第二实现(依赖 collective)。

**P2 —— 多节点横向扩展**
- [ ] Cluster 多节点 init + NodeProbe;`Placement`/`Worker` 跨节点(Flexible + send/recv + NodeAffinity)。
- [ ] `scheduler/manager/`(路由 + 设备/端口锁)。
- [ ] `configs/scheduler/` 组(`ClusterConfig`/`NodeGroupConfig`)+ `start_ray.sh`/`check_ray.sh` + 早校验。
- [ ] replay 作共享/分片服务。

**P3 —— 重型 / 条件项**
- [ ] `scheduler/{hardware,dynamic_scheduler}/`;liger/flash kernel。
- [ ] Channel key 路由 / 加权批;reward/critic worker。
- [ ] **条件**:`hybrid_engines/megatron/`(大模型 TP/PP)、`vllm`/`sglang`(策略变自回归 LLM)——
      仅当 DreamerVLA 模型规模/形态改变才对齐,否则维持非目标。

---

## 附:对齐目标 vs 单机 parity 基线

`OnlineCotrainPipelineRunner`(单机 torchrun,已 merge)仍是 ray backend 的**功能基线与 parity 对照**:
两者共享 `OnlineReplay` / 更新步 / `_rollout_action` / `BaseRunner` 约定。**直接对齐 RLinf 的同时,
每一步都以单机基线做训练等价回归**——既向 RLinf 的分布式/显存能力看齐,又不破坏单机主线行为(P0 的 parity 即守此门)。
RLinf 是 ray-first、无单机回退;DreamerVLA 选择"对齐 RLinf 的能力与组织,但保留单机可退"——这是唯一一处**有意保留的定位差异**(§1)。
