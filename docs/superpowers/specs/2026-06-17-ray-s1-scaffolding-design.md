# 设计(S1):Ray scaffolding —— `dreamervla/scheduler/` 原语(单机最小化,仿 RLinf)

- 日期:2026-06-17
- 状态:子项目 spec,待 review(第 1/5 份)
- 主题:落地总览 §7 的 **S1**:把 RLinf `scheduler/` 的 5 个原语 —— `Cluster` / `Worker` /
  `WorkerGroup` / `Channel` / `Placement` —— 以**单机最小子集**移植进 `dreamervla/scheduler/`,
  **不含任何 DreamerVLA 业务逻辑**(env/infer/learner/replay 是 S2–S4)。
- 范围:**仅 S1**。产出可独立测试的 Ray 编排骨架 + `ray` 必装依赖。后续子项目在其上搭 worker。
- 关联:
  - 总览:`docs/superpowers/specs/2026-06-17-ray-online-cotrain-backend-design.md`(§6 接口契约、§7 拆分)
  - `pyproject.toml` / `requirements.txt`(新增 `ray[default]>=2.47.0` 必装依赖)
  - `tests/unit_tests/`(纯逻辑单测)、`tests/e2e_tests/`(起真 ray)
- 参考实现(RLinf,只取单机子集,砍掉集群机制):
  - `RLinf/rlinf/scheduler/cluster/cluster.py`(`Cluster`)
  - `RLinf/rlinf/scheduler/worker/worker.py`、`worker_group.py`(`Worker` / `WorkerGroup`)
  - `RLinf/rlinf/scheduler/channel/channel.py`(`Channel`)
  - `RLinf/rlinf/scheduler/placement/{placement,packed,node}.py`(`Placement` + 策略)

---

## 1. 范围与边界

S1 **只**产出 `dreamervla/scheduler/` 下 5 个原语,让"起集群 → 按 placement 拉起 N 个同类
actor → 跨 actor 广播方法调用 / 收 `ObjectRef` → actor 间用 channel 传 tensor"这条链路**单机**
跑通且可测。

- **范围内**:`cluster.py` / `worker.py` / `worker_group.py` / `channel.py` / `placement.py`;
  `ray[default]>=2.47.0` 必装依赖;placement 纯函数单测 + 一个最小真 ray smoke(完整/真实测试推 S5,见 §5)。
- **范围外**(明确不做,留给后续子项目):任何 DreamerVLA worker 实现(S2–S4)、`WeightSyncer`
  (S4)、config 组 `configs/scheduler/`(S5)、runner(S5);RLinf 的 collective/NCCL、distributed
  channel、多节点 placement、manager actor、dynamic scheduler、nsight/distributed-log(见 §7)。

---

## 2. 目标 / 非目标

**目标**

1. 5 个原语**单机可用**,API 名称对齐 RLinf(`Cluster` / `Worker` / `WorkerGroup` / `Channel` /
   `Placement` / `PackedPlacementStrategy` / `NodePlacementStrategy`)。
2. **接口 multi-node-clean**:签名不写死单节点;多节点是后续**追加实现**(§7),不改调用方。
3. `ray[default]>=2.47.0` 为**必装依赖**;单机 torchrun 路径默认**不调 `ray.init()`**(不平白起集群),
   故运行时零开销 —— 即"lazy **起集群**",非"lazy import"。
4. **真实测试集中到 S5**(真 ray + 真 worker + parity/训练等价);S1 只留 placement 纯函数单测 + 一个
   最小真 ray smoke 证明骨架本身能跑,不堆 stub/mock。

**非目标**

- 不做 GPU collective / NCCL 权重通信(S4 用 object-store;NCCL 是更后续)。
- 不做 distributed/per-node channel、key 路由、weight 批(只做单 actor FIFO + 计数 `get_batch(n)`)。
- 不做多节点 placement、node group、affinity 调度。
- 不做 config 驱动(原语接收显式参数;config 组在 S5)。

---

## 3. 模块与 API(签名级 + 行为 + 砍掉的 RLinf 机制)

> 约定:device 解析、`CUDA_VISIBLE_DEVICES` 隔离由 `Placement` 决定并经 actor 的 `runtime_env`
> 环境变量注入(`RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `CUDA_VISIBLE_DEVICES`),`Worker.__init__`
> 读取之 —— 与 RLinf 同思路,但只保留单机所需的几个变量。

### 3.1 `scheduler/cluster.py`

```python
class Cluster:                                   # 进程内单例
    def __init__(self, cfg: DictConfig | None = None) -> None: ...
        # if not ray.is_initialized(): ray.init(namespace="DreamerVLA", ...)
        # assert version.parse(ray.__version__) >= version.parse("2.47.0")
    @classmethod
    def has_initialized(cls) -> bool: ...
    @classmethod
    def find_free_port(cls) -> int: ...
    @property
    def num_gpus(self) -> int: ...               # int(ray.cluster_resources().get("GPU", 0))
    @property
    def num_nodes(self) -> int: ...              # 单机恒为 1
    def shutdown(self) -> None: ...              # ray.shutdown()
```
- **砍掉**:namespace 冲突重试、manager actor、跨节点 `NodeProbe`/`NodeGroupInfo`、distributed-log、
  nsight、code-sync runtime_env。单机直接 `ray.init` + 资源查询即可。

### 3.2 `scheduler/placement.py`

```python
@dataclass
class Placement:
    rank: int
    local_rank: int
    local_world_size: int
    visible_accelerators: list[str]              # 写入 CUDA_VISIBLE_DEVICES;CPU worker 为 []
    device: str                                  # "cuda:{local}" 或 "cpu"

class PlacementStrategy(ABC):
    def get_placement(self, cluster: Cluster) -> list[Placement]: ...

class PackedPlacementStrategy(PlacementStrategy):     # GPU worker(InferenceWorker / LearnerWorker)
    def __init__(self, start_gpu: int, end_gpu: int, num_gpus_per_worker: int = 1) -> None: ...
    # 在 [start_gpu, end_gpu] 上顺序打包出 N 个 rank,每 rank 占 num_gpus_per_worker 块 GPU

class NodePlacementStrategy(PlacementStrategy):       # CPU worker(EnvWorker)
    def __init__(self, count: int) -> None: ...
    # 单机产出 count 个无 GPU 亲和的 CPU rank
```
- **砍掉**:`cluster_node_rank` / `placement_node_rank` / `node_group_label` / `MultiNodeGroupResolver` /
  `FlexiblePlacementStrategy`(单机用不到;后续多节点再补)。
- **纯逻辑**:`get_placement` 给定 `cluster.num_gpus` 输出确定的 `Placement` 列表 —— **可不起 ray 单测**
  (把 `cluster` 换成只暴露 `num_gpus` 的轻 stub)。

### 3.3 `scheduler/worker.py`

```python
class Worker:                                    # Ray actor 基类
    def __init__(self) -> None: ...
        # self.rank/local_rank/world_size = int(os.environ[...]);
        # self.device = "cuda:0" if visible else "cpu"  (CUDA_VISIBLE_DEVICES 已被 placement 隔离)
    def init(self) -> None: ...                  # 子类重写:重资源初始化(模型/env);默认 no-op
    @classmethod
    def create_group(cls, *args, **kwargs) -> "WorkerGroup": ...   # 便捷工厂,等价 WorkerGroup(cls, ...)
```
- 子类(S2–S4 的 `EnvWorker` / `InferenceWorker` / `LearnerWorker` / `ReplayWorker`)继承之,
  在 `init()` 里做重初始化、加自己的 RPC 方法。
- **砍掉**:`send/recv/broadcast/send_tensor`(collective)、`manager_proxy`、`device_lock`、
  `create_channel/connect_channel`(改由 `Channel.create/connect` 直连)、timing 装饰器(可后补)。

### 3.4 `scheduler/worker_group.py`

```python
class WorkerGroup:
    def __init__(self, worker_cls: type[Worker], *args, **kwargs) -> None: ...
    def launch(self, cluster: Cluster, placement: PlacementStrategy,
               name: str | None = None) -> "WorkerGroup": ...
        # placement.get_placement(cluster) -> 每 rank 一个 ray actor:
        #   ray.remote(worker_cls).options(num_gpus=..., runtime_env={env_vars: RANK/LOCAL_RANK/
        #   WORLD_SIZE/CUDA_VISIBLE_DEVICES}).remote(*args, **kwargs);随后 .init.remote() 全体
    def execute_on(self, *ranks: int) -> "WorkerGroup": ...   # 只对下一次调用生效,之后复位
    def __getattr__(self, method: str) -> "WorkerGroupFunc": ...   # 把子类方法名广播成组调用

class WorkerGroupFunc:
    def __call__(self, *args, **kwargs) -> "WorkerGroupFuncResult": ...   # 向(被 execute_on 过滤的)各 actor 发 method.remote

class WorkerGroupFuncResult:
    def wait(self) -> list[Any]: ...             # ray.get 全体
    def done(self) -> bool: ...                  # ray.wait(timeout=0)
```
- **砍掉**:`from_group_name`/全局注册表、`max_concurrency`/`isolate_gpu`/`catch_system_failure`/
  `disable_distributed_log` 等旋钮、`async_wait`、`consume_duration*`、`NodeAffinitySchedulingStrategy`。
  保留 `execute_on` 子集执行(主 runner 重叠调度会用,如"只让 learner rank 干活")。

### 3.5 `scheduler/channel.py`

```python
@ray.remote
class _ChannelActor:                             # 单 actor 背书 FIFO
    def put(self, item: Any) -> None: ...
    def get(self) -> Any: ...                    # 空则阻塞/轮询
    def get_batch(self, n: int) -> list[Any]: ...
    def qsize(self) -> int: ...

class Channel:
    @classmethod
    def create(cls, name: str, maxsize: int = 0) -> "Channel": ...   # 起一个具名 _ChannelActor
    @classmethod
    def connect(cls, name: str) -> "Channel": ...                    # 连已存在的具名 actor
    def put(self, item: Any) -> None: ...
    def get(self) -> Any: ...
    def get_batch(self, n: int) -> list[Any]: ...                    # gather k 路(env→infer)
    def qsize(self) -> int: ...
    def empty(self) -> bool: ...
```
- 同步语义(`ray.get` 阻塞)即可满足 S1;**异步 `async_op`**、双缓冲重叠留到 S5 调度层按需补。
- **砍掉**:`distributed`/per-node、`key` 路由、`weight` 批、worker-context 内的 `send/recv` 快路径、
  `local`(纯 in-process)。统一走"单具名 ray actor FIFO";`get_batch` 按**计数 n**,不按 weight。

---

## 4. 依赖(ray 必装)

- `pyproject.toml` 把 `ray[default]>=2.47.0` 列为**普通(必装)依赖**;`requirements.txt` 同步。
  → **本机当前未装,需先 `pip install "ray[default]>=2.47.0"`**,否则真 ray smoke 与后续子项目跑不了。
- 既然 ray 必装,**不做** import 隔离 / lazy import(`dreamervla.scheduler` 可自由 import)。
- **唯一保留的纪律**:单机 torchrun 路径**默认不调 `ray.init()`**(不平白起一个 ray 集群);
  只有选到 Ray backend 才 `Cluster()` 起集群 —— 即"lazy **起集群**"。这条由 S5 的 backend 分支守。

---

## 5. 测试 / 验收

> 原则(用户定):**真实测试集中到 S5**。S1 不堆 stub/mock ray 的单测,只留:一个 placement 纯函数单测
> + 一个最小真 ray smoke,证明编排骨架本身能跑、不是日后 S5 失败的来源。

1. **`test_scheduler_placement.py`(纯函数,CI 默认跑)** —— 测真实逻辑、不 mock ray:
   - `PackedPlacementStrategy(0, 3).get_placement(cluster(num_gpus=4))` → 4 个 rank,`device` 依次
     `cuda:0..3`、`local_world_size=4`、`visible_accelerators` 正确。
   - `num_gpus_per_worker=2`、`start/end` 子区间、GPU 不足时报错(早校验)。
   - `NodePlacementStrategy(3)` → 3 个 CPU rank(`device=="cpu"`、`visible==[]`)。
2. **`test_scheduler_ray_smoke.py`(真 ray;ray 必装故 CI 默认跑)**:
   - `Cluster()` 起本地 ray;`num_gpus`/`num_nodes` 查询成功;重复 `Cluster()` 幂等。
   - dummy `Worker` 子类(`init` 置标志、`echo(x)->x`、`add(a,b)->a+b`),`WorkerGroup` 按
     `NodePlacementStrategy(N)` 拉起 N 个 → 广播 `echo`/`add`,`wait()` 收齐 N 个;`execute_on(0)` 只回 1 个。
   - `Channel.create` → 一个 actor `put` k 条、另一个 actor `get_batch(k)` 收齐(顺序/内容正确,含小 tensor)。
   - `cluster.shutdown()` 收尾。

**推到 S5 的真实测试**:InferenceWorker parity、env→replay 一致、weight-sync 正确、端到端训练与单机
基线等价 + 可观测重叠 —— 全部用真 worker/真负载做;S1–S4 的中间产物在 S5 一并真实验证。

**验收**:1+2 全过;`ruff`/类型检查过;单机路径运行时不起 ray 集群。

---

## 6. 实现顺序(TDD)

1. `placement.py` + `test_scheduler_placement.py`(纯逻辑,先红后绿)。
2. `cluster.py`(ray.init + 资源查询 + 版本断言)。
3. `worker.py`(env 读取 + `init` 生命周期 + `create_group`)。
4. `worker_group.py`(launch/广播/`execute_on`/result)。
5. `channel.py`(`_ChannelActor` + `Channel`)。
6. 声明 `ray[default]>=2.47.0` 必装依赖(`pyproject.toml` / `requirements.txt`)。
7. `test_scheduler_ray_smoke.py` 串起 2–5(最小真 ray smoke;parity/训练等价等真实测试在 S5)。

---

## 7. 留给后续子项目 / 多节点(接口已预留)

- **多节点 placement**:补 `cluster_node_rank` / node group / `FlexiblePlacementStrategy`;`get_placement`
  签名不变。
- **collective / NCCL**:`Worker.send/recv/broadcast` + collective group —— S4 的 NCCL `WeightSyncer` 用。
- **distributed channel**:per-node actor、key 路由、weight 批、async 快路径 —— 吞吐 profile 需要时再上。
- **config 驱动**:`configs/scheduler/`(GPU 分配、worker 数、`channel_maxsize`)在 S5。
- **runner / 重叠调度**:S5。

---

## 8. 风险

- **`get` 阻塞实现**:`_ChannelActor.get` 空队列时的阻塞/轮询会占 actor 线程。S1 用简单轮询 +
  `max_concurrency` 或 asyncio actor;先正确后调优(吞吐留 S5)。
- **`runtime_env` 注入 env 变量的可移植性**:不同 ray 版本注入方式略异;以 `≥2.47` 为准并在 e2e 守住。
- **单机平白起集群**:单机路径若误调 `Cluster()`/`ray.init()` 会平白起一个 ray 集群占资源;
  runner backend 分支须确保只有选到 Ray 才起集群(S5 守)。
- **单例 `Cluster`**:进程内多次 `Cluster()` 必须幂等(`ray.is_initialized` 守),否则重复 `ray.init` 报错。
