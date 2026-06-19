# Ray 后端对齐报告:DreamerVLA 按 RLinf 手动挡哲学补齐

- 日期:2026-06-19(重定向:由"差异对比"改为"按 RLinf 手动挡哲学补齐能力"的现状 + 方案)
- 目标:让 DreamerVLA 的可选 Ray backend 对齐相邻 RLinf 仓库的工程架构、训练闭环和
  **手动资源杠杆**——不再把 RLinf 的分布式 / 显存优化栈当作"刻意省略",而是逐项列为
  **待对齐的补齐目标**并给出方案与优先级。
- 基准:`RLinf`(`main`)的 `rlinf/scheduler/`、`rlinf/workers/`、`rlinf/hybrid_engines/`、
  `rlinf/runners/`、`ray_utils/`、`pyproject.toml`。
- 现状:`DreamerVLA`(`main`)的 `dreamervla/{scheduler,workers,hybrid_engines,runners}/`、
  `configs/experiment/*ray*`、`tests/{unit_tests,e2e_tests}`。
- 设计依据(本仓):`docs/superpowers/specs/2026-06-17-ray-online-cotrain-backend-design.md` 及 S1–S6 spec(gitignored)。

> **一句话**:DreamerVLA 已经按 RLinf 命名/分层落好了**骨架**(`scheduler`/`worker`/`worker_group`/`channel`/
> `placement`、`workers/{env,inference,actor,replay}`、`hybrid_engines/weight_syncer`、RLinf 风格 runner)。
> **本次已补齐核心单机 Ray 对齐项**:真实 DreamerVLA learner phase、手动 AMP/FSDP/offload/checkpointing
> 杠杆、collective weight-sync 接口、hardware discovery、FlexiblePlacement、model registry 与手动 config groups。
> **仍需后续推进的是横向扩展项**:manager 层、多节点 node 管理、dynamic scheduler、S5 单机 parity 容差测试、
> bucket/patch/压缩和条件性的 Megatron/vLLM/SGLang。
>
> ⚠️ **对齐边界提醒**:RLinf **本身也不做"自动适应显存"**——没有 auto-batch-size、没有 OOM-retry、不用
> DeepSpeed/ColossalAI/Lightning。它提供的是**手动 config 驱动**的显存优化构件。所以"对齐 RLinf 的显存能力"
> 指的是**移植这套手动栈**,**不是**去造一个 VRAM 自适应器(见 §4)。batch size、micro-batch、
> gradient accumulation、并行 env 数、FSDP/offload/precision/checkpointing/kernel 选择都应由 recipe 显式给出,
> 系统只负责校验组合、记录资源指标和暴露可调杠杆。

---

## 0. 对齐目标与现状速览

| 维度 | RLinf(对齐基准) | DreamerVLA 现状 | 对齐动作 | 优先级 |
|---|---|---|---|---|
| Ray 依赖定位 | **核心必装** `ray[default]>=2.47.0` | **opt-in extra** `[ray]`,import 隔离 | 决策点:是否升核心(§1,默认保留 opt-in) | — |
| `Cluster` | 多节点单例 + NodeProbe + manager 启动 | 单机幂等单例(`num_nodes=1`) | 加多节点 init / 节点发现 / manager 拉起 | P2 |
| `Worker`/`WorkerGroup` | actor 基类 + 组广播 + send/recv 路由 | 同形(广播/`execute_on`/`ray.get`) | 补点对点 `send/recv` + NodeAffinity 调度 | P2 |
| `Placement` | Packed/Node/**Flexible** + 范围语法 + 跨节点 bundle | Packed/Node(单机,ray-free 可单测) | 补 Flexible + 范围语法 + 多节点 bundle | P2 |
| `Channel` | actor FIFO + **key 路由 + 加权批** + AsyncWork | actor FIFO + `get_batch(n)` | 补 key 路由 / 加权批 / 统一异步句柄 | P3 |
| **Collective / NCCL** | `collective/`(NCCL/Gloo,多通道,broadcast/send/recv) | ✅ `scheduler/collective/` torch broadcast helper(未初始化 dist 时本地 no-op) | 继续补多通道/send/recv | P1 |
| **Manager 层** | Worker/Collective/Node/Lock manager(named actor) | ❌ 无 | 新建 `scheduler/manager/`(多节点协调/锁) | P2 |
| **多节点 / Node 管理** | NodeProbe + 节点组 + 跨机 env/解释器 | ❌ localhost only | NodeProbe + 节点组 config + bootstrap | P2 |
| **Hardware 注册/发现** | GPU/NPU/机器人注册表与设备枚举 | ✅ `scheduler/hardware.py` CUDA 发现/校验(不自动改 batch/env) | 后续扩 NPU/机器人注册表 | P3 |
| **Dynamic scheduler** | 组件级预取/流水线编排 | ❌ runner 内手动 `ObjectRef` 重叠 | 新建 `scheduler/dynamic_scheduler/` | P3 |
| **显存:分片(ZeRO 等价)** | FSDP(FSDP1+2)/ Megatron(TP/PP/SP) | ✅ `hybrid_engines/fsdp/FSDPModelManager` 已接 learner | 后续补 FSDP2/Megatron 条件项 | **P0/P1** |
| **显存:CPU offload** | FSDP `CPUOffloadPolicy` + 优化器 offload | ✅ `FSDPModelManager.cpu_offload` | 继续补优化器 offload 细项 | P1 |
| **显存:混合精度** | AMP autocast+GradScaler / FSDP MixedPrecision | ✅ learner precision + AMP/GradScaler + FSDP MixedPrecision | 继续补更多 runner 指标 | P0/P1 |
| **显存:激活重计算** | `gradient_checkpointing_enable` / Megatron recompute | ✅ `activation_checkpointing` 调用 `gradient_checkpointing_enable` | Megatron recompute 保持条件项 | P1 |
| **显存:高效 kernel** | liger_kernel / FlashAttention | ❌ | 可选接(默认关,由 config 手动选择) | P3 |
| 权重同步 | NCCL 默认 + bucket/patch/压缩 | ✅ object-store + `CollectiveWeightSyncer.broadcast_model` | 后续补 bucket/patch/压缩 | P1 |
| Workers 角色 | actor/env/rollout/**reward/critic** | env/inference/replay/actor(单卡) | 补 reward/critic worker(若 RL 需要) | P3 |
| Runner | `EmbodiedRunner`(infer→step→learn) | ✅ `OnlineCotrainRayRunner` 可调 `dreamervla_cotrain` 真实 phase;S5 parity 待闭合 | 补单机 parity 容差测试 | **P0** |
| ray_utils / 启动 | `start_ray.sh`/`check_ray.sh`(head/worker 跨机) | 无(`Cluster` 内 `ray.init`) | 多节点时补启动脚本 | P2 |
| cluster 配置 | `ClusterConfig`/`NodeGroupConfig` | experiment YAML 扁平 | 引入 `scheduler/` config 组 + 早校验 | P2 |
| vLLM / SGLang / Megatron | ✅ `hybrid_engines/{vllm,megatron}` | ❌ | **条件对齐**:仅当策略变自回归 LLM / 上大模型 | P3(条件) |

> 优先级口径:**P0 = 阻塞一切的训练正确性**;**P1 = 单机即可见收益的核心对齐(显存栈 + NCCL)**;
> **P2 = 多节点横向扩展**;**P3 = 重型/条件项**。

### 0.1 非目标:不做 VRAM 自适应库

这份报告替代"探测显存 -> 自动调 batch/env 数 -> 自动避 OOM"的方向。按 RLinf 对齐时,下面这些都不是目标:

- 不新增 `training.auto_vram_batch` / `collect.auto_vram_envs`。
- 不做两步 probe/back-calculate,不把 `0.85 * total_vram` 作为 runtime 填充目标。
- 不在训练启动时自动改 `dataloader.batch_size`、`gradient_accumulate_every` 或 rollout `envs_per_gpu`。
- 不做 OOM retry 后自动降档。

应当补的是**手动杠杆 + 早校验 + 可观测指标**:operator 明确选择 batch/env/FSDP/offload/precision/checkpointing/kernel
组合,`validate_cfg` 在启动前校验世界大小、batch 可整除、placement 和资源声明是否一致,runner 记录固定配置下的显存峰值和吞吐,
供下一次人工调参使用。

---

## 1. 依赖与定位(对齐的第一个决策点)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 声明 | core `dependencies`:`ray[default]>=2.47.0`(scheduler 的基石) | `[project.optional-dependencies] ray`;基础依赖仅 `hydra-core`/`omegaconf` |
| 隔离 | 无(全框架 ray-first) | `scheduler/__init__.py` 禁止单机 torchrun 路径 import 本包 |

**如果完全复制 RLinf 的产品定位**,会要求把 ray 提升为**核心必装依赖**(与 RLinf 一致),去掉 import 隔离。

> **建议(需用户拍板)**:**默认保留 opt-in**。RLinf 是 ray-first 分布式框架,ray 当核心理所当然;
> DreamerVLA 是单机训练框架 + 可选 ray 扩展,把 ray 升核心会让纯单机用户被迫装 ray 全家桶、失去"单机零 ray 开销"。
> 工程组织按 RLinf 补齐,但依赖定位保持 DreamerVLA 主线不变:Ray 是可选后端,不是默认训练拓扑。仅当多节点成为主线时再升核心。

---

## 2. Scheduler 原语对齐

> 骨架已对齐(`Worker`/`WorkerGroup`/`Channel`/`Placement` 形态与公共 API 几乎一一对应,`Placement` 抽成
> ray-free 模块反而比 RLinf 更干净)。下面只列**补齐项**。

### 2.1 Cluster(P2)
- RLinf:`rlinf/scheduler/cluster/cluster.py` —— `ray.init(address="auto")`、`NodeProbe` 远程发现每节点、
  在 rank-0 拉起 Worker/Collective/Node/Lock manager、SIGUSR1 优雅退出、`get_alive_nodes/get_node_info/get_node_ip`。
- DVLA:`dreamervla/scheduler/cluster.py` —— 单机幂等 `ray.init`、`num_nodes` 硬编码 1、版本断言、`find_free_port`。
- **对齐动作**:`num_nodes` 改为从 NodeProbe / cluster metadata 获取;加 `address="auto"` 连接已起集群;新增 NodeProbe(见 §3 多节点)+ manager 启动钩子。

### 2.2 Worker / WorkerGroup(P2,基本已齐)
- RLinf 多了 `send/recv/send_tensor` 点对点(经 WorkerManager 路由)+ `NodeAffinitySchedulingStrategy` 硬绑节点。
- **对齐动作**:在 `worker.py` 加点对点 `send/recv`(多节点时用,单机仍走 `Channel`);`worker_group.launch` 支持
  `NodeAffinitySchedulingStrategy`(依赖 §3 节点信息)。

### 2.3 Placement(P2)
- **对齐动作**:`placement.py` 补 `FlexiblePlacementStrategy` + 范围语法(`"0-3,5,7-9"`)+ 跨节点 bundle(`(node_rank, gpu_bundle)`)。

### 2.4 Channel(P3)
- **对齐动作**:`channel.py` 补 key 路由、加权批(`get_batch(weight=...)`)、统一 `AsyncWork` 句柄;非阻塞 `put_no_wait/get_no_wait`。

### 2.5 RLinf 有、DVLA 无的 scheduler 子系统(补齐方案)

| 子系统 | RLinf 提供 | 对齐动作 | 优先级 |
|---|---|---|---|
| `collective/`(NCCL/Gloo) | `Collective`/`CollectiveGroup`/`MultiChannelProcessGroup`:`broadcast/send/recv`、多通道、ring broadcast、CPU↔object-store / GPU↔NCCL | 新建 `dreamervla/scheduler/collective/`;先支撑 NCCL 权重同步(§5),再支撑多卡 learner | **P1** |
| `manager/` | Worker/Collective/Node/DeviceLock/PortLock manager(named actor 全局元数据/路由/锁) | 新建 `dreamervla/scheduler/manager/`;单机可空实现,多节点接 send/recv 路由 + 设备锁 | P2 |
| `cluster/node.py` 多节点 | NodeProbe 远程发现、节点组、跨机 env_vars/解释器、Nsight 包裹 | 接入 §2.1 Cluster;新增节点组 config(§7) | P2 |
| `hardware/` | GPU(N/AMD/Intel)/Ascend NPU/机器人注册表,设备枚举排序 | 新建 `dreamervla/scheduler/hardware/`;最小先做 NVIDIA 枚举,用于 placement/校验,不用于自动调 batch/env | P3 |
| `dynamic_scheduler/` | `ComponentManager` 组件级预取/流水线编排 | 新建 `dreamervla/scheduler/dynamic_scheduler/`;把 runner 内手写的 `ObjectRef` 重叠抽成调度器 | P3 |

---

## 3. Workers 对齐

| RLinf `rlinf/workers/` | DreamerVLA 现状 | 对齐动作 |
|---|---|---|
| `env/`(`EnvWorker`/`AsyncEnvWorker`) | `env/env_worker.py`(每 env 一 actor,done 自动 reset) | **已齐**;可选补 `AsyncEnvWorker` |
| `rollout/`(`MultiStepRolloutWorker`/`SGLangWorker`) | `inference/{inference_worker,rollout_inference_worker}.py`(encoder+WM+policy 同 actor) | **已齐**(刻意不拆推理链);`SGLangWorker` 仅自回归 LLM 才需(条件) |
| `data`(replay) | `replay/replay_worker.py`(包 `OnlineReplay`) | **已齐** |
| `actor/`(FSDP 训练:`learn/get_weight/set_weight`) | `actor/learner_worker.py`(**单卡裸 Adam,synthetic PPO**) | **核心差距**:① 接真实更新步(§6);② 接 FSDP 显存栈(§4) |
| `reward/`(`EmbodiedRewardWorker`) | ❌(outcome reward 在 env/算法内) | 若需独立 reward 服务再补(当前 outcome 模式不需要) |
| `critic/`(FSDP 价值网络) | ❌(critic 不参与动作选择,推理不传) | 若 RL 需独立 critic worker 再补 |

---

## 4. 手动显存优化栈对齐(一等维度,P0/P1)

> 这是按 RLinf 补齐时**单机即可见收益**的核心一块。RLinf 的显存优化是**手动 config 驱动**的整套构件;
> DVLA 的 ray `LearnerWorker` 目前**一样都没有**(单卡、裸 Adam、synthetic PPO)。
> **再次强调对齐边界**:RLinf 没有 auto-batch-size / OOM-retry / DeepSpeed/Lightning,所以这里的"对齐"
> = **移植手动栈**,batch size / micro-batch / env 数仍由 config 手填。

| 手动杠杆 | RLinf 实现(基准) | 对齐动作(DVLA) | 优先级 |
|---|---|---|---|
| 参数/梯度/优化器分片(ZeRO 等价) | `hybrid_engines/fsdp/`(`fsdp.py`=FSDP1、`fsdp2.py`=FSDP2、`fsdp_model_manager.py`、`strategy/`) | 新建 `dreamervla/hybrid_engines/fsdp/`,提供 `FSDPModelManager` 包 learner 的 policy/WM;`worker_group` 已能多卡分配,补 DDP/FSDP 初始化;由 config 显式选择策略 | **P1** |
| CPU offload | FSDP `CPUOffloadPolicy` + `optimizer_cpu_offload`/`offload_fraction` | 随 FSDP 一并接;config `learner.fsdp.cpu_offload` 等显式开关 | P1 |
| 混合精度(AMP) | `amp_autocast`(autocast+`GradScaler`)/ FSDP `MixedPrecision`(param/reduce/buffer dtype);**默认关** | learner update 包 `torch.autocast` + `GradScaler`;config `learner.precision=bf16|fp16|fp32` | **P0/P1** |
| 激活重计算 | `module.gradient_checkpointing_enable()`(FSDP 路径)/ Megatron `recompute_granularity/method/num_layers` | 给 WM/policy 模块加 `gradient_checkpointing_enable`;config 开关 | P1 |
| 高效 kernel | `use_liger_kernel`(默认 False)/ FlashAttention(`attention_backend="FLASH_ATTN"`) | 可选接,默认关(与 RLinf 一致),由 config 显式开启 | P3 |
| 张量/流水并行 | `hybrid_engines/megatron/`(TP/PP/SP + microbatch + sequence packing + 优化器 offload) | **条件对齐**:仅当单卡放不下、需大模型并行时引入;TP/PP/SP 拓扑显式写入 config | P3(条件) |

**落点建议**:在 `dreamervla/hybrid_engines/` 下镜像 RLinf 的 `fsdp/` 子树(`FSDPModelManager` + `strategy/{base,fsdp,fsdp2,checkpoint}.py`),
`LearnerWorker.init()` 用它包装 policy/WM;`learner.train_cfg` 暴露 `fsdp/precision/cpu_offload/activation_checkpointing` 开关并入 `validate_cfg` 早校验。
这里的早校验只验证"人工给定的组合是否自洽":world size、placement、micro-batch、global batch、gradient accumulation、FSDP shard
策略和 dtype 是否匹配;它不替用户搜索这些值。

---

## 5. 权重同步对齐(P1)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 接口 | `WeightSyncer`(ABC)+ `BucketWeightSyncer`/`PatchSyncer`/`Compressor` | `WeightSyncer`(ABC,`push/pull`) |
| 传输 | **NCCL collective 默认**(ring broadcast)+ object-store(CPU/异构回落) | **仅** object-store(`_WeightStore` actor,单调版本,CPU `state_dict`) |

**对齐动作**:沿现有 `WeightSyncer` ABC **加 NCCL 第二实现**(依赖 §2.5 `collective/`):learner→inference 走 broadcast,
object-store 作 CPU/异构回落;可选 bucket 分桶。接口已预留,**追加实现即可、无需重写**。

---

## 6. Runner / 训练闭环对齐(P0,最高优先)

| | RLinf(基准) | DreamerVLA 现状 |
|---|---|---|
| 主循环 | `EmbodiedRunner`:init Cluster → launch group → Channel/WeightSyncer 协调 → 收指标 | `OnlineCotrainRayRunner`:**形态已对齐**(infer→step→learn 重叠、`time/overlap_events`) |
| 学习步 | 真实 PPO/SAC/NFT 更新(FSDP) | **`LearnerWorker.update` = synthetic PPO(对 mean action 做 MSE)** |

**对齐动作(阻塞项)**:
1. 把 `LearnerWorker.update` 从 synthetic PPO 换成**真实 DreamerVLA RL / WM / classifier 更新步**(复用单机 `online_dreamervla.py` 的更新逻辑)。
2. 补 **S5 训练等价 parity**:同 config/seed 下,ray backend 的 loss/指标对齐单机 `OnlineCotrainPipelineRunner` 到容差内。
3. 产出**可观测重叠证据**(`time/` 命名空间下 infer/learner 并发占比),否则 ray 仅是复杂化(设计 §12)。

> 此项不补,§4/§5 的显存与 NCCL 对齐都是在错误的训练上加速,**必须先行**。

---

## 7. ray_utils / 启动 / config 对齐(P2)

| | RLinf(基准) | 对齐动作 |
|---|---|---|
| 启动脚本 | `ray_utils/start_ray.sh`(rank0 `ray start --head`,worker 轮询 head IP 后 `--address`)、`check_ray.sh`(轮询 `ray status` 等 GPU) | 多节点阶段补 `scripts/start_ray.sh`/`check_ray.sh`;单机仍由 `Cluster` 内 `ray.init` 管 |
| cluster 配置 | `ClusterConfig`/`NodeGroupConfig`:`num_nodes`、`component_placement`、`node_groups`(跨机硬件/env/解释器)、`nsight` | 引入 `configs/scheduler/` 组(对齐 RLinf 的 cluster 配置);`validate_cfg` 校验节点组/placement/GPU 数 |

---

## 8. 测试对齐(随各项推进)

现状:7 单测契约(scheduler/placement/worker 公共 API,ray-free)+ 6 e2e(真 ray smoke/parity,`test_s1..s6`)。
**对齐动作**:每补一项加配套测试——FSDP/AMP 显存栈接入后补**固定 config 下**的显存峰值回归(只观测/防回退,不驱动自动调参);
NCCL `WeightSyncer` 接入后补 object-store/NCCL 双实现 parity;
真实更新步接入后把 `test_s5_*` 从 synthetic 升级为**对单机基线的训练等价 parity**;多节点接入后补跨节点 smoke。

---

## 9. 命名对齐表(RLinf → DreamerVLA;"待建"= 补齐目标落点)

| RLinf 路径 | DreamerVLA 路径 | 状态 |
|---|---|---|
| `rlinf/scheduler/cluster/cluster.py` | `dreamervla/scheduler/cluster.py` | 已齐(单机)→ 补多节点 |
| `rlinf/scheduler/worker/{worker,worker_group}.py` | `dreamervla/scheduler/{worker,worker_group}.py` | 已齐 → 补 send/recv |
| `PlacementStrategy`(worker 子系统内) | `dreamervla/scheduler/placement.py`(独立) | 已齐 → 补 Flexible/多节点 |
| `rlinf/scheduler/channel/channel.py` | `dreamervla/scheduler/channel.py` | 已齐 → 补 key 路由/加权批 |
| `rlinf/scheduler/collective/` | `dreamervla/scheduler/collective/` | 已建基础 broadcast helper → 补多通道/send/recv |
| `rlinf/scheduler/manager/` | `dreamervla/scheduler/manager/` | **待建(P2)** |
| `rlinf/scheduler/hardware/` | `dreamervla/scheduler/hardware.py` | 已建 CUDA discovery → 补 NPU/机器人注册表 |
| `rlinf/scheduler/dynamic_scheduler/` | `dreamervla/scheduler/dynamic_scheduler/` | **待建(P3)** |
| `rlinf/workers/env/` | `dreamervla/workers/env/env_worker.py` | 已齐 |
| `rlinf/workers/rollout/` | `dreamervla/workers/inference/{inference_worker,rollout_inference_worker}.py` | 已齐 |
| `rlinf/workers/actor/`(FSDP) | `dreamervla/workers/actor/learner_worker.py` | 骨架齐 → 接真实更新步 + FSDP |
| `rlinf/workers/{reward,critic}/` | `dreamervla/workers/{reward,critic}/` | 待建(条件) |
| `rlinf/hybrid_engines/weight_syncer/`(NCCL) | `dreamervla/hybrid_engines/weight_syncer/`(object-store) | 已齐 → 补 NCCL 实现 |
| `rlinf/hybrid_engines/fsdp/` | `dreamervla/hybrid_engines/fsdp/` | 已建 `FSDPModelManager` → 补 FSDP2/strategy 子树 |
| `rlinf/hybrid_engines/{megatron,vllm}/` | — | 条件对齐(P3) |
| `rlinf/runners/embodied_runner.py` | `dreamervla/runners/online_cotrain_ray_runner.py` | 形态齐 → 接真实更新步 |
| `ray_utils/{start_ray,check_ray}.sh` | `scripts/{start_ray,check_ray}.sh` | 待建(P2) |

---

## 10. 对齐路线图(按优先级)

**P0 —— 训练正确性(阻塞一切)**
- [x] `LearnerWorker.update`:synthetic PPO → 真实 DreamerVLA RL/WM/classifier 更新步(复用 `online_dreamervla.py`)。
      已新增 `mode=dreamervla_cotrain`,内置 `wm` / `classifier` / `rl` / `cotrain` phase,并保留
      `synthetic_ppo` 作为 cheap smoke 模式。
- [ ] S5 训练等价 parity:对单机 `OnlineCotrainPipelineRunner`,同 seed loss/指标对齐到容差。
- [x] 可观测重叠证据(`time/` 并发占比)。
- [x] learner 接混合精度(AMP autocast + GradScaler / dtype 策略),由 recipe 显式选择 dtype。

**P1 —— 单机可见收益的核心对齐**
- [x] `hybrid_engines/fsdp/`:镜像 RLinf `FSDPModelManager`,learner 接 FSDP 分片 + CPU offload + 激活重计算,全部由 config 手动开启。
- [x] `scheduler/collective/`:NCCL/Gloo collective group。
- [x] `WeightSyncer` NCCL 第二实现(依赖 collective)。

**P2 —— 多节点横向扩展**
- [ ] Cluster 多节点 init + NodeProbe;`Placement`/`Worker` 跨节点(Flexible + send/recv + NodeAffinity)。
- [ ] `scheduler/manager/`(路由 + 设备/端口锁)。
- [ ] `configs/scheduler/` 组(`ClusterConfig`/`NodeGroupConfig`)+ `start_ray.sh`/`check_ray.sh` + 早校验。
- [ ] replay 作共享/分片服务。

**P3 —— 重型 / 条件项**
- [x] `scheduler/hardware/`:CUDA 设备发现/校验,不服务自动扩 batch/env。
- [ ] `scheduler/dynamic_scheduler/`;liger/flash kernel。
- [ ] Channel key 路由 / 加权批;reward/critic worker。
- [ ] **条件**:`hybrid_engines/megatron/`(大模型 TP/PP)、`vllm`/`sglang`(策略变自回归 LLM)——
      仅当 DreamerVLA 模型规模/形态改变才对齐,否则维持非目标。

**横切 —— 配置 / 模型解耦(见 §11,独立于 ray 分布式,可单独落地)**
- [x] precision 归一化下沉 config-time(§11.3)——`validate_cfg` 校验 `learner.train_cfg.precision`。
- [x] 模型注册表 `dreamervla/models/registry.py`(§11.1)——中成本、高收益,解模型↔yaml 硬耦合。
- [x] `configs/{parallelism,precision}/` 组(§11.4,落实 §0.1 手动杠杆)。
- [ ] 更大范围的 config dataclass 化(§11.2)。

---

## 11. Hydra 配置与模型解耦:借鉴 RLinf 的改进点

> 用户单独点名的方向。RLinf 在"模型↔配置解耦"和"配置早校验/归一化"上比 DreamerVLA 更系统化。
> 下面是**与 ray 分布式无关、单独就能落地**的改进项,按收益/成本排。

### 11.1 模型注册表(headline:模型解耦)——中成本、高收益

- **RLinf**:`rlinf/models/__init__.py` 有中央注册表 `_MODEL_REGISTRY` + `register_model(model_type, builder)` +
  `get_model(cfg)`(按 `cfg.model_type` 派发到 builder);`SupportedModel` 枚举在实例化前校验类型已注册,未知类型直接报错并列出支持清单。
- **DreamerVLA 现状(其实已经走了一半)**:`dreamervla/models/embodiment/{openvla,openvla_oft,...}/__init__.py` **已各有
  `get_model(cfg, torch_dtype)` builder**(与 RLinf builder 同形),但**没有中央注册表**;runner 走
  `hydra.utils.instantiate(cfg.policy)`(yaml 里写全类路径 `_target_`),ray worker 走 `_build_from_cfg`
  (`importlib` 解析 `target` 串)。模型类挪位置 → 所有引用它的 yaml 的 `_target_` 都要手改。
- **改进**:新建 `dreamervla/models/registry.py`,把**已有的 `get_model` builder 注册进去**,按 `model_type` 派发 +
  启动期校验"类型已注册";配置从 `_target_: 全类路径` 迁到 `model_type + kwargs`,**保留 `_target_` 回落**以渐进迁移。
- **收益**:模型重构不再牵动一堆 yaml;有清晰模型清单;启动期就报"未知 model_type"而不是 worker 里炸。**成本中等、收益高,建议优先。**

### 11.2 配置 dataclass 化 + 高价值早校验——中成本

- **RLinf**:cluster/node/parallelism 用 `@dataclass`(`ClusterConfig`/`NodeGroupConfig`/…)+ `__post_init__` 早归一化;
  `validate_cfg`(`rlinf/config.py:1205`)按任务分派(`validate_embodied_cfg`/`validate_fsdp_cfg`/`validate_megatron_cfg`),
  **worker spawn 前** fail-fast(40+ 断言:并行整除、vocab 对齐、loss↔model 兼容)。
- **DreamerVLA 现状**:裸 `DictConfig` + 轻量 `validate_cfg`(`dreamervla/config.py`,已有 `_validate_logger_backends`/
  `_validate_training_batch`/`_validate_chunk_horizon_consistency`/`_validate_ray_manual_resources` 等,但**只查不变式、不做类型归一**)。
- **改进(取中庸,别照搬 1500 行)**:关键 config 引 dataclass + `__post_init__`(precision str→枚举、int 强制);
  补**高价值早校验**:`task_type`↔runner 一致、`model_type` 已注册(配合 11.1)、`resume` 路径存在、`chunk≤seq_len`、
  FSDP/precision 组合合法(配合 §4)。保留 DVLA 的轻量灵活,只补高价值早检。

### 11.3 precision 归一化下沉到 config-time——低成本(与 §4 AMP 配套)

- **RLinf**:`torch_dtype_from_precision()`(`rlinf/config.py:142`)在 **config-build 时**把 `"bf16"/"fp16"/16/"16-mixed"` 等转 `torch.dtype`。
- **DreamerVLA 现状**:在 **worker init 时**才于 `learner_worker._resolve_precision()` 归一化,每个 worker 各算一遍、无早校验。
- **改进**:加 `dreamervla/config.py::torch_dtype_from_precision`,`validate_cfg` 时算好缓存,worker 直接读。
  单一真相源;且 §4 给 learner 接 AMP 时正好复用。**最便宜的一项,建议和 §4 P0 的 AMP 一起做。**

### 11.4 config group 按关注点拆分——低/中成本(落实 §0.1 的"手动杠杆")

- **RLinf**:`model/`、`training_backend/`(`fsdp`/`megatron`)、`weight_syncer/` 是**独立可组合**的 config group,
  用 package 描述符(`@actor.model`、`@actor.fsdp_config`)路由进 cfg 树;experiment 用 `override /...` 切换。
- **DreamerVLA 现状**:组按**任务**分(`VLA/`/`worldmodel/`/`classifier/`),precision/parallelism/dataloader 散在各处。
- **改进**:新增 `configs/{parallelism,precision}/` 组(与 §4 显存栈天然对齐:`parallelism/fsdp.yaml`、`precision/bf16.yaml`),
  experiment 用 `override /precision: fp32` 之类组合。**把 §4 的手动显存杠杆做成可组合 config group,正是 §0.1"手动杠杆 + 早校验 + 可观测"的落地形式。**

### 11.5 (条件)Megatron/transformer config 派生——低优先

- **RLinf**:`build_transformer_config`(`config.py:1312`)/`_build_model_parallel_config`(`config.py:1508`)把 cfg 映射成
  Megatron 的 typed config + 早断言(`vocab % tp_size` 等)。
- **DreamerVLA**:用 embodied 单卡模型,暂不需要;仅当上 Megatron 才对齐(同 §4 的 Megatron 条件项)。

> **runner 选择**:DVLA 用 `_target_`(`train.py:41` `hydra.utils.get_class(cfg._target_)`),RLinf 按 task_type + 类注册。
> 两者都干净,**不建议改**;`PUBLIC_RUNNERS` 留作清单/校验即可。

---

## 12. 被废弃的自动显存计划

`docs/superpowers/plans/2026-06-19-vram-autosize.md` 的方向应视为废弃:它试图实现 probe + 反推 micro-batch/env 数,
这属于 DeepSpeed/Lightning/ColossalAI 式"自动挡"体验,不是 RLinf 的工程哲学。若后续需要计划文件,应改写成
"手动显存杠杆接入计划":FSDP/CPU offload/precision/activation checkpointing/kernel/placement/config validation/metrics,
而不是 `auto_vram_*`。

---

## 附:对齐目标 vs 单机 parity 基线

`OnlineCotrainPipelineRunner`(单机 torchrun,已 merge)仍是 ray backend 的**功能基线与 parity 对照**:
两者共享 `OnlineReplay` / 更新步 / `_rollout_action` / `BaseRunner` 约定。**按 RLinf 补齐能力的同时,
每一步都以单机基线做训练等价回归**——既向 RLinf 的分布式/显存能力看齐,又不破坏单机主线行为(P0 的 parity 即守此门)。
RLinf 是 ray-first、无单机回退;DreamerVLA 选择"对齐 RLinf 的能力与组织,但保留单机可退"——这是唯一一处**有意保留的定位差异**(§1)。
