# Ray 对齐:已实现(DreamerVLA 单机 RLinf 对齐)

- 日期:2026-06-19
- 用途:记录 DreamerVLA 可选 Ray backend **已经按 RLinf 工程组织落地**的部分(含代码落点),
  以及对齐时确立的**设计立场 / 通信模型 / 工程经验**(evergreen 参考)。
- 配套:待实现项见 **`docs/ray_rlinf_alignment_todo.md`**;对齐基准是相邻 `RLinf` 仓库。

> **一句话**:单机 Ray 对齐的**核心已完成**——RLinf 风格 scheduler 骨架、env/infer/replay/learner workers、
> 真实 DreamerVLA learner 训练闭环、手动显存栈(FSDP/AMP/offload/激活重计算)、collective 权重同步、
> 模型注册表与手动 config groups 均已在仓。剩下的是横向扩展(多节点/manager/dynamic scheduler)
> 与少数扩展/条件项,见 todo 文档。

---

## 1. 已实现清单(按子系统,带代码落点)

### 1.1 Scheduler 骨架(`dreamervla/scheduler/`)
- `cluster.py` —— 幂等单机 `ray.init`(`namespace="DreamerVLA"`、`include_dashboard=False`、loopback)、版本断言 ≥2.47、
  `find_free_port`、`num_gpus`、`num_nodes`/`require_single_node`、`shutdown`。
- `worker.py` / `worker_group.py` —— Worker 基类(env 注入 rank/device)+ WorkerGroup(组广播、`execute_on(ranks)`、
  `send/recv` 单 rank helper、`WorkerGroupFuncResult.wait()/done()`);多 rank 组会注入同一个本机
  `MASTER_ADDR`/`MASTER_PORT`,供单节点 FSDP/Gloo/NCCL rendezvous。
- `placement.py` —— `PackedPlacementStrategy`/`NodePlacementStrategy`/**`FlexiblePlacementStrategy`** + **范围语法 `parse_accelerator_range`**(`"0-3,5,7-9"`),ray-free 可单测。
- `channel.py` —— actor 背书 FIFO(`create/connect/put/get/get_batch(n)`),支持 key 路由和 weighted batch,
  detached named actor。
- `node.py` —— `NodeInfo` + `discover_ray_nodes`(节点元数据发现,单机=1)。
- `manager/` —— ray-free `WorkerManager` route table + `DeviceLockManager` 最小单机协调原语。
- `dynamic_scheduler.py` —— executor-backed `ComponentScheduler` 最小组件重叠调度原语。
- `hardware.py` —— **CUDA 设备发现 / 校验**(服务 placement / 早校验,**不**自动改 batch/env)。
- `collective/` —— torch broadcast helper(未初始化 dist 时本地 no-op),为 NCCL 权重同步打底。

### 1.2 Workers(`dreamervla/workers/`)
- `env/env_worker.py` —— 每 env 一 actor,done 自动 reset,step 内灌 `replay.add_episode`。
- `inference/inference_worker.py` —— encoder+world_model+policy 同 actor 批量前向(刻意不拆推理链)。
- `inference/rollout_inference_worker.py` —— OFT 冷启动采集,per-env action queue + 夹爪后处理。
- `replay/replay_worker.py` —— 包 `OnlineReplay`(`add_episode/sample/size/ready/task_stats`)。
- `actor/learner_worker.py` —— 见 §1.3。

### 1.3 Learner 训练闭环(`actor/learner_worker.py` + `runners/online_cotrain_ray_runner.py`)
- **真实 DreamerVLA 更新步**:新增 `mode=dreamervla_cotrain`,内置 `wm` / `classifier` / `rl` / `cotrain` phase
  (复用 `online_dreamervla.py` 更新逻辑);保留 `synthetic_ppo` 作 cheap smoke。
- `OnlineCotrainRayRunner`(`runner_name="online_cotrain_ray"`):infer→step→learn 重叠主循环,
  **可观测重叠证据**(`time/overlap_events` 等);learner placement 可由 config 选择 `node`/`packed`/`flexible`,
  `learner.train_cfg.device=auto` 在 GPU actor 内解析为 local `cuda:0`。

### 1.4 手动显存优化栈(`dreamervla/hybrid_engines/fsdp/`)
- `fsdp_model_manager.py::FSDPModelManager` —— learner 接 **FSDP 分片 + `cpu_offload` + 激活重计算
  (`gradient_checkpointing_enable`)**,全部由 config 显式开启。
- 单节点多卡:当 `WORLD_SIZE>1` 且 `strategy=fsdp` 时,FSDP manager 会按 Ray actor env 初始化
  `torch.distributed` process group;`WORLD_SIZE=1` 保持 no-op。
- **混合精度**:learner update 包 `torch.autocast` + `GradScaler`,FSDP `MixedPrecision`;`learner.train_cfg.precision`(bf16/fp16/fp32)。
- 边界:这些都是**手动杠杆**,batch/micro-batch/env 数仍由 recipe 手填;系统只校验组合自洽 + 记指标(见 §3.2)。

### 1.5 权重同步(`dreamervla/hybrid_engines/weight_syncer/`)
- object-store 默认实现(`_WeightStore` actor,单调版本,CPU `state_dict`,payload=`world_model`+`policy`)。
- **`collective.py::CollectiveWeightSyncer.broadcast_model`** —— NCCL/collective 第二实现(依赖 `scheduler/collective/`)。

### 1.6 配置 / 模型解耦
- **模型注册表** `dreamervla/models/registry.py`(+ `models/__init__.py`):`register_model`/`get_model` 按 `model_type` 派发,
  解模型 ↔ yaml `_target_` 硬耦合(保留 `_target_` 回落)。
- **precision 归一化下沉 config-time**:`dreamervla/config.py::torch_dtype_from_precision` + `validate_cfg` 校验 `learner.train_cfg.precision`。
- **手动 config groups**:新增 `configs/parallelism/`、`configs/precision/`、`configs/scheduler/`
  (落实 §3.1 的"手动杠杆 + 早校验")。
- **ops scripts**:`scripts/start_ray.sh` / `scripts/check_ray.sh` 用于本地 Ray head/status 调试;正式 runner 仍可由
  `Cluster` 内部自动 `ray.init`。

### 1.7 测试
- 单测契约(ray-free):`tests/unit_tests/test_scheduler_*`、`test_ray_*`(placement/worker 公共 API、依赖声明、coldstart config/adapter)。
- e2e(真 ray smoke/parity):`tests/e2e_tests/test_s{1..6}_*`(WorkerGroup、ReplayWorker、InferenceWorker、LearnerWorker+sync、cotrain runner、coldstart collect)。

---

## 2. 已建命名映射(RLinf → DreamerVLA)

| RLinf 路径 | DreamerVLA 路径 | 已实现部分 |
|---|---|---|
| `scheduler/cluster/cluster.py` | `scheduler/cluster.py` | 单机 bootstrap |
| `scheduler/worker/{worker,worker_group}.py` | `scheduler/{worker,worker_group}.py` | 基类 + 组广播 + execute_on |
| `PlacementStrategy` | `scheduler/placement.py` | Packed/Node/**Flexible** + 范围语法 |
| `scheduler/channel/channel.py` | `scheduler/channel.py` | actor FIFO + get_batch |
| `scheduler/cluster/node.py` | `scheduler/node.py` | NodeInfo + discover(单机) |
| `scheduler/hardware/` | `scheduler/hardware.py` | CUDA 发现/校验 |
| `scheduler/collective/` | `scheduler/collective/` | broadcast helper |
| `workers/{env,rollout,data}` | `workers/{env,inference,replay}` | 已齐 |
| `workers/actor/` | `workers/actor/learner_worker.py` | 真实 phase + FSDP |
| `hybrid_engines/fsdp/` | `hybrid_engines/fsdp/fsdp_model_manager.py` | FSDPModelManager |
| `hybrid_engines/weight_syncer/` | `hybrid_engines/weight_syncer/{__init__,collective}.py` | object-store + collective |
| `runners/embodied_runner.py` | `runners/online_cotrain_ray_runner.py` | infer→step→learn |
| `rlinf/models/__init__.py`(registry) | `dreamervla/models/registry.py` | model registry |

---

## 3. 设计立场与边界(evergreen)

### 3.1 非目标:不做 VRAM 自适应库
按 RLinf 对齐时**刻意不做**"探测显存 → 自动调 batch/env → 自动避 OOM":
- 不新增 `training.auto_vram_batch` / `collect.auto_vram_envs`;不做 probe + 反推;不把 `0.85*total_vram` 当 runtime 填充目标;不做 OOM-retry 自动降档。
- RLinf 本身也没有 auto-batch-size / OOM-retry / DeepSpeed/Lightning。对齐 = **移植手动 config 栈**。
- 系统职责:暴露杠杆(batch/env/FSDP/offload/precision/checkpointing/kernel)+ `validate_cfg` 启动前校验组合自洽 + runner 记录显存峰值/吞吐供人工调参。
- (历史)`docs/superpowers/plans/2026-06-19-vram-autosize.md` 方向已废弃,理由同上。

### 3.2 依赖定位:Ray 保持 opt-in(有意保留的差异)
- RLinf:`ray[default]>=2.47.0` 为**核心必装**(ray-first 框架)。
- DreamerVLA:`[project.optional-dependencies] ray`,基础依赖仅 `hydra-core`/`omegaconf`;`scheduler/__init__.py` **禁止单机 torchrun 路径 import 本包**。
- 立场:工程组织按 RLinf 补齐,但 ray 是**可选后端、不是默认拓扑**;纯单机用户零 ray 开销。仅当多节点成为主线时再考虑升核心。
- 单机 parity 基线:`OnlineCotrainPipelineRunner`(单机 torchrun)是 ray backend 的功能基线与训练等价对照(parity 测试见 todo P0)。

---

## 4. Ray 通信模型与端口(单机一定要"通信"吗?)

**结论**:Ray 并行模型 = 多 OS 进程(actor 各独立进程),**进程间通信内生绕不开**;但**单机全走 loopback(127.0.0.1)+ 共享内存,不碰外网/防火墙端口**。真正需要网络/跨机端口的只有**多节点**(见 todo P2)。

| 组件 | 作用 | 通信 |
|---|---|---|
| GCS | 控制面元数据 | 本地 TCP(gRPC) |
| raylet | 本节点调度 | 本地端口 |
| Plasma object store | 大 tensor 按引用传 | **共享内存 `/dev/shm`** + socket(非 TCP) |
| dashboard(已关) | 监控 | HTTP 端口(本仓 `include_dashboard=False`) |
| worker/actor | 跑代码 | `f.remote()` 走 gRPC 本地端口;数据走 object store |

要点:大数据走共享内存不走 TCP;TCP 端口仅控制面/RPC 且 Ray 自动选空闲端口;本仓 `Cluster` 绑 loopback、关 dashboard、`find_free_port` 不写死端口;**单机 torchrun 默认路径不调 `ray.init()` → 零 ray 进程/端口**。
想再少开端口:`local_mode=True`(本仓 cfg 可透传)在 driver 进程内串行跑、几乎无端口,但**没有真并行**(仅调试,新版 Ray 弃用);GCS/raylet/object-store 端口正常模式去不掉。
**实务**:单机跑 ray backend 不用配网络/开防火墙;端口/网络只在多节点才是真问题。

---

## 5. 值得借鉴的工程经验

1. **object-store-first + 有界队列解耦**:大 payload 按引用进共享内存,`Channel`(actor FIFO)给 env/infer/learn 做速率解耦 + 背压,队列即同步点,不靠手写锁/线程。
2. **命名 detached actor 做服务发现**(本仓 Channel/WeightStore 已用):给单例服务起名字,别层层传 handle。
3. **共置资源显式仲裁(DeviceLock/PortLock)**:多 actor 共卡/抢端口时用 lock manager 仲裁(本仓 P2 manager 层)。
4. **spawn 前 fail-fast 全图校验**:`validate_cfg` 在起 worker 前校验整张配置图——两小时的任务不该在第 3 个 worker 里因拼写错误才崩。
5. **手动杠杆 + 可观测,拒绝 auto-magic**:暴露每个旋钮并记 `train/ env/ rollout/ time/` 指标,而不是把性能藏进自动调参。
6. **幂等启动 + 确定性退出**:`ray.is_initialized()` 守护、namespace 防撞、信号 → `ray.kill` 清理。
7. **`runtime_env`/`py_modules` 同步代码到远端 worker**:多节点可复现性靠把代码当成要分发的状态(本仓多节点阶段再用)。
8. **控制面 / 数据面分离**:编排走 `WorkerGroup` 广播,大数据走 `Channel`/object store。
9. **(本仓已做对、要守住)opt-in 不侵入**:ray import 隔离 + 单机 torchrun 默认——这是 RLinf(ray-first、无单机回退)做不到的优势,别为了对齐丢掉。
