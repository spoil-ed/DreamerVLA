# Ray 对齐:待实现(DreamerVLA → RLinf 剩余 TODO)

- 日期:2026-06-19
- 用途:把 DreamerVLA 可选 Ray backend 对齐 RLinf **尚未完成**的工作汇总成一份单一 TODO。
- 已完成的部分与设计立场见 **`docs/ray_rlinf_alignment_implemented.md`**;对齐基准是相邻 `RLinf` 仓库。

> **现状**:单机核心对齐已完成(scheduler 骨架、workers、真实 learner 闭环、FSDP/AMP/offload 显存栈、
> collective 权重同步、模型注册表、手动 config groups)。**剩余收敛为**:① 一个阻塞性 P0(单机训练等价 parity 测试);
> ② 几项"已建基础、待补全"的扩展;③ 多节点横向扩展(P2);④ 少数重型/条件项(P3)。
>
> 优先级:**P0 = 阻塞训练正确性** · **P1 = 单机扩展(已有基础)** · **P2 = 多节点** · **P3 = 重型/条件**。

---

## P0 —— 阻塞项(必须先行)

### [ ] S5 训练等价 parity 测试
- **动作**:对单机 `OnlineCotrainPipelineRunner`,同 config/seed 下让 ray backend(`mode=dreamervla_cotrain`)的 loss/指标
  对齐到容差内;把 `tests/e2e_tests/test_s5_*` 从 synthetic 升级为**对单机基线的训练等价 parity**。
- **依赖**:真实 learner phase(已实现)。
- **验收**:同 seed 下 ray vs 单机的关键 loss/指标 `allclose` 到约定容差;CI 可跑(真 ray)。
- **理由**:真实更新步已接,但**等价性尚未被测试守住**;不闭合则下面的显存/NCCL 加速都可能是在错误训练上加速。

---

## P1 —— 单机扩展(已有基础,待补全)

### [ ] collective 多通道 / 点对点 send/recv
- **现状**:`scheduler/collective/` 仅 torch broadcast helper(未初始化 dist 时本地 no-op)。
- **动作**:补多通道(`MultiChannelProcessGroup` 形)+ `send/recv`,对齐 RLinf `rlinf/scheduler/collective/`。
- **验收**:单机多卡下 broadcast/send/recv 正确;为多卡 learner 与 NCCL 权重同步打底。

### [ ] 权重同步 bucket / patch / 压缩
- **现状**:object-store + `CollectiveWeightSyncer.broadcast_model` 已有。
- **动作**:沿 `WeightSyncer` ABC 补 RLinf 的 `BucketWeightSyncer`/`PatchSyncer`/`Compressor`(分桶/增量/压缩)。
- **验收**:大模型权重同步带宽/延迟下降;与现有实现 parity。

### [ ] FSDP2 + strategy 子树
- **现状**:`hybrid_engines/fsdp/FSDPModelManager` 已接 learner(FSDP 分片 + offload + 激活重计算)。
- **动作**:补 RLinf 的 `strategy/{base,fsdp,fsdp2,checkpoint}.py` 子树(FSDP2 路径 + 可插拔策略)。
- **验收**:FSDP1/FSDP2 可由 config 切换;checkpoint 保存/恢复正确。

### [ ] 更大范围的 config dataclass 化
- **现状**:precision 已下沉 config-time;`validate_cfg` 有若干 `_validate_*`。
- **动作**:关键 config 引 `@dataclass` + `__post_init__`(类型归一)+ 补高价值早校验(`task_type`↔runner、`model_type` 已注册、`resume` 路径存在、`chunk≤seq_len`、FSDP/precision 组合合法)。
- **边界**:取中庸,**不照搬** RLinf 的 ~1500 行;只补高价值早检,保留 DVLA 轻量灵活。

---

## P2 —— 多节点横向扩展(端口/网络从这里才真正相关)

### [ ] Cluster 多节点 init + 跨节点 NodeProbe
- **动作**:`Cluster` 加 `address="auto"` 连接已起集群;`num_nodes` 从 cluster metadata 取;`node.py` 扩成跨机 NodeProbe(远程发现每节点硬件/env/解释器)。
- **验收**:两节点起一个 Ray 集群,`discover_ray_nodes` 报 2 节点。

### [ ] Placement / Worker 跨节点
- **动作**:`placement.py` 跨节点 bundle;`worker.py`/`WorkerManager` 补跨节点点对点 `send/recv`;`worker_group.launch` 支持 `NodeAffinitySchedulingStrategy` 硬绑节点。
- **验收**:worker 按节点组放置;跨节点 send/recv 正确。

### [ ] `scheduler/manager/` 多节点化(全局协调 + 锁)
- **动作**:把当前 ray-free manager 层扩成 named actor(Worker/Collective/Node manager + **DeviceLock/PortLock**),做全局元数据/路由/资源仲裁。
- **验收**:多 worker 共卡/抢端口时由 lock manager 仲裁,无冲突。

### [ ] cluster 配置 + 多节点启动模式
- **动作**:扩展现有 `configs/scheduler/` 组(对齐 RLinf `ClusterConfig`/`NodeGroupConfig`:`num_nodes`/`component_placement`/`node_groups`)+ `validate_cfg` 校验节点组/placement/GPU 数;把现有 `scripts/start_ray.sh` 从本地 head 调试扩成 head/worker bootstrap,`check_ray.sh` 扩成轮询 GPU/节点健康。
- **验收**:一条命令拉起多节点集群并通过早校验。

### [ ] replay 作共享 / 分片服务
- **动作**:`ReplayWorker` 升级为跨节点共享或分片的 replay 服务。
- **验收**:多节点 worker 共享同一 replay,采样/插入一致。

---

## P3 —— 重型 / 条件项

### [ ] `scheduler/dynamic_scheduler/` 升级为 RLinf 式调度器
- 把当前 executor-backed 最小 `ComponentScheduler` 升级为 RLinf 式 `ComponentManager` 组件级预取/流水线编排。
- (含原 H3:`cold_start_ray_collect_runner._run_loop_overlap` 升级为 RLinf sync-pipeline parity——推理跑 batch t 时上批 env-step 在飞、预取下帧 obs,仅 dump-size 达标时阻塞;不采用完整 `AsyncEmbodiedRunner` 并发。)

### [ ] Channel async API + reward/critic worker
- `channel.py` 已有 key 路由和 weighted batch;后续补统一 `AsyncWork` 句柄、`put_no_wait/get_no_wait`。
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

- 端口/网络只在 **P2 多节点**才相关;单机所有 P0/P1/P3 项都只用 loopback + 共享内存(见 implemented 文档 §4)。
- 所有显存/资源项均为**手动杠杆 + 早校验 + 可观测**,**不做 VRAM 自适应**(见 implemented 文档 §3.1)。
- 每补一项加配套测试;真实更新步相关回归优先(P0)。
