# 设计(总览):Ray 在线 cotrain backend(仿照 RLinf,单机多 GPU 先行)

- 日期:2026-06-17
- 状态:总览设计,待 review;通过后再依次写 5 份子项目 spec,全部 spec 完成后才实现
- 主题:把 §10 记录的"Ray 整体在线 loop"从纸面选项落成**可选 distributed backend**:
  把在线 cotrain loop(infer → env step → learner)拆成异构 Ray worker(InferenceWorker /
  EnvWorker / LearnerWorker / ReplayWorker),做 infer→step→learn 流水线重叠,仿照
  `RLinf/rlinf/` 的工程组织。**单机多 GPU 先行,接口 multi-node-clean。**
- 性质:本文是**总览**(锁定 worker 映射、关键决策、子项目拆分、接口契约);各子项目的
  详细 spec 另写,见 §7。
- 关联(本仓):
  - `dreamervla/runners/online_cotrain_pipeline_runner.py`(`runner_name="online_cotrain_pipeline"`,
    当前单机 torchrun loop,本 backend 的等价基线)
  - `dreamervla/runners/online_cotrain_runner.py`(在线机制母体)
  - `dreamervla/runners/online_replay.py`(`OnlineReplay`)
  - `dreamervla/runners/online_cotrain_runner.py:184-217`(`_rollout_action`,InferenceWorker 复用的在线前向路径;encoder 冻结 + world_model + policy)
  - `dreamervla/runners/vec_rollout_env.py`(env 步进逻辑)
  - `dreamervla/runners/online_dreamervla.py`(`online_classifier_update_step` 等更新步)
  - `dreamervla/runners/base_runner.py`(`BaseRunner`:run-artifact / checkpoint / logging 约定)
  - `dreamervla/config.py`(`validate_cfg`,早校验)
  - `dreamervla/runners/__init__.py`(`PUBLIC_RUNNERS` 注册)
- 参考实现(RLinf,只学组织,不照搬拓扑):
  - `RLinf/rlinf/scheduler/cluster/cluster.py`(Ray init / 节点探测)
  - `RLinf/rlinf/scheduler/worker/worker.py`、`worker_group.py`(Worker 基类 + WorkerGroup)
  - `RLinf/rlinf/scheduler/placement/placement.py`(component→GPU bundle 映射)
  - `RLinf/rlinf/scheduler/channel/channel.py`(FIFO 异步队列)
  - `RLinf/rlinf/workers/env/env_worker.py`、`workers/inference/`、`workers/actor/`
  - `RLinf/rlinf/runners/embodied_runner.py`(infer→step→learn 主循环)
  - `RLinf/rlinf/hybrid_engines/weight_syncer/`(权重同步抽象)

---

## 1. 背景与动机

当前 `OnlineCotrainPipelineRunner` 是**单机 torchrun/DDP** 的在线 loop:同一进程(每个
rank)里串行跑"采 rollout → 灌 `OnlineReplay` → 训 WM/classifier + RL 训 policy",env
步进用 `SubprocVecEnv` 子进程并行。`§10`(`2026-06-16-rlinf-vectorized-rollout-migration.md`)
把"Ray 整体 loop"记成**未来架构选项**:异构 worker 放置、infer-step-learner 重叠、多节点
replay 服务,但本期不实现、不引入依赖。

本设计把该选项落地为一条**与单机 loop 并存的可选 backend**。动机:让 GPU 推理、CPU env
步进、learner 反传三者**重叠**以掩盖互等;并为将来 multi-node 横向扩展铺好接口。

> **策略已定(本轮交付的边界)**:Ray 是 **opt-in backend**,单机 torchrun 仍是默认主线;
> 单机多 GPU 先行,接口 multi-node-clean;不引入 vLLM / Megatron;weight sync 先 object-store。
> 详见 §3。

---

## 2. 目标 / 非目标

**目标**

1. 提供一条可选的 Ray 在线 cotrain backend,与单机 `OnlineCotrainPipelineRunner` **行为等价**
   (相同 config/seed 下训练结果可对齐),但把 infer/env-step/learner 摆成异构 Ray worker 并重叠。
2. **最大复用**现有组件代码:InferenceWorker 复用 `_rollout_action` 前向;EnvWorker 复用
   `vec_rollout_env.py` 单 env 步进;ReplayWorker 复用 `OnlineReplay`;LearnerWorker 复用 WM/
   classifier/RL 更新步。新代码集中在 Ray 编排层(cluster / worker / channel / placement /
   weight_sync)与一个新 runner。
3. 所有编排抽象 **multi-node-clean**:placement / channel / weight-sync 接口不写死单节点假设,
   将来加多节点是**追加子项目**,不是重写。
4. 沿用 `BaseRunner` 的 run-artifact / checkpoint / logging 约定与 `train/ eval/ env/
   rollout/ time/` 命名空间;`validate_cfg` 早校验新增 Ray/placement 字段。
5. 每个子项目带低成本 smoke,并对关键路径做 parity 测试(对照单机基线)。

**非目标**

- **不替换**单机 loop(单机仍默认、不删);Ray 仅 opt-in。
- **不引入 vLLM / SGLang**:DreamerVLA 推理是定长前向产 action chunk,非自回归 token 生成。
- **不引入 Megatron**:模型单卡可放,learner 用现有 DDP/FSDP。
- **本轮不做真多节点**:不搭 head/worker 跨机网络、不做 NCCL 跨机 collective、不做共享 FS
  replay 服务——这些是后续子项目(§11)。
- 不改单机 `OnlineCotrainPipelineRunner` / `OnlineCotrainRunner` 的纯在线行为(只做机械抽取
  以便组件被 Ray worker 复用;若需抽取,抽取本身要 parity)。

---

## 3. 关键决策(已与用户确认)

| # | 决策 | 理由 |
|---|---|---|
| D1 | Ray 作为**可选 backend**,单机 torchrun 仍默认 | 低 blast radius;两条路共享组件代码 |
| D2 | **单机多 GPU 先行**,接口 multi-node-clean | 今天可测、RLinf 形状;网络复杂度后置为独立子项目 |
| D3 | **EnvWorker = 每 env 一个 Ray actor**,丢掉 `SubprocVecEnv` | Ray actor 本身即进程隔离,替代子进程层;别嵌套双层并行 |
| D4 | **InferenceWorker 不拆**(VLA encoder + WM 前向 + policy 同一 actor) | 推理链串行,拆开只增每帧序列化跳数、无并行收益;单卡放得下 |
| D5 | **Learner / EnvWorker 才是并发重叠点** | 反传(GPU)与 env 步进(CPU)和推理(GPU)算力性质不同,可真重叠 |
| D6 | **不要 vLLM、不要 Megatron** | 见 §2 非目标 |
| D7 | weight sync **先 object-store `state_dict`**,接口预留 NCCL broadcast | 单机、K 步一次,够用;先搭接口、性能后置 |
| D8 | 拆成 **5 个子项目,各一份 spec**;**全部设计完再实现** | 总览给共享上下文,逐份 review 早期抓接口偏差 |

---

## 4. Worker 映射(RLinf → DreamerVLA)

| RLinf 组件 | DreamerVLA Ray 等价 | 复用现有代码 |
|---|---|---|
| `scheduler/cluster/` | `Cluster`:单机 `ray.init(namespace="DreamerVLA")`,版本断言 ≥2.47 | 新,薄 |
| `scheduler/worker/`(Worker + WorkerGroup) | `Worker` 基类 + `WorkerGroup`(`num_gpus` 分配) | 新 |
| `scheduler/placement/` | 单机 placement:InferenceWorker→GPU0、Learner→GPU1.._、EnvWorker×k→CPU | 新,最小 |
| `scheduler/channel/` | `Channel`:Ray actor 背书的 FIFO 队列(env↔infer↔replay↔learner) | 新 |
| `workers/inference/` | `InferenceWorker`:VLA encoder + WM 前向 + policy → action chunk + action-hidden | `_rollout_action` |
| `workers/env/` | `EnvWorker`:单 LIBERO env 步进的 Ray actor | `vec_rollout_env.py` |
| `workers/actor/`(learner) | `LearnerWorker`:WM / classifier / RL 更新步 | `online_classifier_update_step` 等 + DreamerVLA RL |
| `data/`(replay) | `ReplayWorker`:`OnlineReplay` 包成 Ray actor(共享 push/pull) | `online_replay.py` |
| `hybrid_engines/weight_syncer/` | `WeightSyncer`:object-store `state_dict`(NCCL 为后续升级) | 新 |
| `runners/embodied_runner.py` | `OnlineCotrainRayRunner`:infer→step→learn 重叠主循环;opt-in;复用 BaseRunner | 仿 `OnlineCotrainPipelineRunner` |

**代码落点(RLinf 命名风格):**

```
dreamervla/
  scheduler/                       # Ray 编排层(对应 rlinf/scheduler/)
    cluster.py                     # Cluster
    worker.py / worker_group.py    # Worker(Ray actor 基类)/ WorkerGroup
    channel.py                     # Channel
    placement.py                   # Placement
  workers/                         # Ray actor 实现(对应 rlinf/workers/)
    env/env_worker.py              # EnvWorker(单 env)
    inference/inference_worker.py  # InferenceWorker(VLA+WM+policy 前向)
    actor/learner_worker.py        # LearnerWorker(WM/classifier/RL 更新)
    replay/replay_worker.py        # ReplayWorker(包 OnlineReplay)
  hybrid_engines/weight_syncer/    # 对应 rlinf/hybrid_engines/weight_syncer/
    base.py / objectstore.py       # WeightSyncer(ABC)/ ObjectStoreWeightSyncer
  runners/
    online_cotrain_ray_runner.py   # OnlineCotrainRayRunner(runner_name="online_cotrain_ray")
```

新 runner 按现有约定加入 `PUBLIC_RUNNERS`。`hybrid_engines/` 仅含 `weight_syncer/`
(fsdp/megatron/vllm 等子目录为非目标,刻意不建)。

---

## 5. 架构与数据流

```
                 ┌─────────── Channel:obs_batch ───────────┐
  EnvWorker×k ───┤                                          ▼
   (CPU,各1env)  │                                    InferenceWorker (GPU0)
        ▲        └──────────── Channel:actions ◀──── VLA+WM+policy 前向
        │                                                   │ 产 step 记录(obs/action/hidden)
        │ env.step(action)                                  ▼
        └────────────────────────────────────────── ReplayWorker (OnlineReplay)
                                                            │ sample batch
        weights(每K步,object-store) ◀── WeightSyncer ─── LearnerWorker (GPU1..)
                                                            │ WM/classifier/RL 更新步
                                                            └─ push 新 policy 权重
```

- **rollout 路径(infer↔env 重叠)**:k 个 EnvWorker 各产 1 obs → `Channel` gather 成 batch →
  InferenceWorker 一次批量前向 → `Channel` scatter 回 k 个 action → 各 EnvWorker `env.step` →
  step 记录写 `ReplayWorker`。auto-reset 在 EnvWorker 内(episode 结束自动重置,沿用现有 env 语义)。
- **learner 路径(与 rollout 重叠)**:LearnerWorker 从 `ReplayWorker` `sample` 出 batch,跑 WM /
  classifier / RL 更新步,每 K 步通过 `WeightSyncer` 把新 policy 权重 push 给 InferenceWorker。
- **重叠调度**:rollout 与 learner 跑在不同 actor/GPU 上,主 runner 用 Ray `ObjectRef` 异步等待
  + 双缓冲让两者重叠(对应 RLinf 的 channel + 异步 handle.wait 模式)。**重叠的正确性**(权重版本、
  replay 读写一致)由各子项目 spec 细化。

---

## 6. 接口契约(multi-node-clean,签名级)

下列为**抽象签名**(锁定边界,细节在各子项目 spec)。要点:不写死"单节点/单 GPU",
节点/设备由 placement 决定,通信走 Channel/WeightSyncer,组件不感知拓扑。

```python
# scheduler/cluster.py
class Cluster:
    def __init__(self, cfg) -> None: ...          # ray.init(namespace=...), assert ray>=2.47, 探测节点
    @property
    def gpus(self) -> int: ...                     # 可用 GPU 数(单机=本机;多节点=全集群)

# scheduler/placement.py
class Placement:                                   # component -> (node, gpu_bundle)
    def assign(self, role: str, count: int) -> list[ResourceBundle]: ...
    # 单机实现:按本机 GPU/CPU 分;多节点实现:跨节点分(后续子项目,接口不变)

# scheduler/worker.py(Worker)+ scheduler/worker_group.py(WorkerGroup)
class Worker:                                      # Ray actor 基类:rank/world_size/device 自动注入
    def init(self) -> None: ...
class WorkerGroup:                                 # 一组同类 actor 的容器,统一 launch / 广播调用
    @classmethod
    def launch(cls, worker_cls, count, placement, name) -> "WorkerGroup": ...
    def call(self, method: str, *a, ranks=None) -> list["ObjectRef"]: ...

# scheduler/channel.py
class Channel:                                     # actor 背书 FIFO;CPU/GPU tensor 均可
    @classmethod
    def create(cls, name: str, maxsize: int) -> "Channel": ...
    def put(self, item) -> "ObjectRef": ...
    def get(self) -> "ObjectRef": ...
    def get_batch(self, n: int) -> "ObjectRef": ...   # gather k 路

# hybrid_engines/weight_syncer/base.py(ABC)+ objectstore.py(默认实现)
class WeightSyncer(ABC):                            # 工厂:object-store 实现 / (后续)NCCL 实现
    def push(self, state_dict, version: int) -> None: ...
    def pull(self, model, version: int) -> None: ...

# workers/{env,inference,actor,replay}/*_worker.py
class EnvWorker(Worker):          def step(self, action) -> StepRecord: ...   # 单 env + auto-reset
class InferenceWorker(Worker):    def forward_batch(self, obs_batch) -> ActionBatch: ...
class ReplayWorker(Worker):       def add(self, records): ...;  def sample(self, bsz): ...
class LearnerWorker(Worker):      def update(self, batch) -> dict: ...;  def policy_state_dict(self): ...
```

> 数据结构(`StepRecord` / `ActionBatch`)对齐 `OnlineReplay` 现有 episode 格式与
> `_rollout_action` 的输入/输出,**避免新增 schema**;具体字段在子项目 #2/#3 spec 核实。

---

## 7. 子项目拆分(各一份 spec,依赖顺序)

| # | 子项目 | 范围 | 依赖 | 验收(verify) |
|---|---|---|---|---|
| S1 | **Ray scaffolding** | `Cluster` + `Worker`/`WorkerGroup` + `Placement`(单机) + `Channel`;无 DreamerVLA 业务逻辑 | — | spawn N 个 dummy actor;tensor 过 channel gather/scatter 正确;smoke 测试 |
| S2 | **ReplayWorker + EnvWorker** | `OnlineReplay` 包成 actor;单 env 步进 actor + auto-reset;env→replay 灌注 | S1 | k 个 EnvWorker 采集的 episode 落 ReplayWorker,格式 == 现 `OnlineReplay` |
| S3 | **InferenceWorker** | VLA+WM+policy 前向 actor;env↔infer rollout 闭环(channel gather/scatter) | S1,S2 | 同 obs+seed 下 action chunk / action-hidden 与现 `_rollout_action` **parity** |
| S4 | **LearnerWorker + WeightSyncer** | WM/classifier/RL 更新步 actor;object-store 权重 push→InferenceWorker | S1,S2 | loss 下降;push 后 InferenceWorker 权重版本/数值正确 |
| S5 | **OnlineCotrainRayRunner + 重叠调度** | 串起全链;infer→step→learn 重叠;config backend 选择;`validate_cfg` 扩展;AGENTS.md 软化 | S1–S4 | smoke config 端到端训练,与单机基线**等价**且有可观测重叠 |

S1–S4 全部按 **multi-node-clean 接口**实现(§6),后续多节点为追加(§11)。

---

## 8. Config 集成

- 新 runner `OnlineCotrainRayRunner`(`runner_name="online_cotrain_ray"`),沿现有 `PUBLIC_RUNNERS`
  注册方式;通过 `experiment=<name>` 选择,**不**新增顶层 route YAML(遵 AGENTS.md/CLAUDE.md)。
- 新增 config 组(拟)`scheduler/`(对应 RLinf 的 cluster/placement 配置;或并入现有组),含:`num_env_workers`、`infer_gpu`、
  `learner_gpus`、`weight_sync_every`、`channel_maxsize`、`placement` 策略名。
- `validate_cfg` 早校验:GPU 数 ≥ 需求、`weight_sync_every` 与 horizon/chunk 一致、env worker 数与
  batch 假设一致、Ray 版本可用。
- 沿用 `logger=tensorboard_wandb` 默认与 `train/ eval/ env/ rollout/ time/` 命名空间。

---

## 9. AGENTS.md 软化

把 `AGENTS.md:6` 与 `AGENTS.md:69-71` 的"无 Ray / 不引入 Ray stack"由**禁止**改为
**"单机 torchrun 为默认主线;Ray 作为 opt-in distributed backend 可用,但不得成为默认、
不得侵入单机路径"**。同步更新 `CLAUDE.md` 的 RLinf Alignment Snapshot。此改动并入 S5。

---

## 10. 测试 / 验收

- 每子项目带 **smoke**(tiny config:1–2 env worker、几步)。
- **Parity 测试**是核心:S3 推理 parity(对 `_rollout_action`)、S5 训练等价(对单机
  `OnlineCotrainPipelineRunner`,同 seed 下 loss/指标对齐到容差内)。
- ray 必装,**真 ray e2e 默认在 CI 跑**;纯函数(placement 分配)可单独快测。**真实测试集中在
  S3(推理 parity)与 S5(训练等价 / 可观测重叠)**,不堆 stub/mock(用户定)。

---

## 11. 非目标 / 未来(本轮不做,接口已预留)

- **真多节点**:head/worker 跨机、replay 作共享/分片服务、跨机 NCCL collective、多机 learner
  DDP/FSDP。`Placement`/`Channel`/`WeightSyncer` 接口已 multi-node-clean,届时为**追加实现**。
- **NCCL broadcast weight sync**:`WeightSyncer` 第二实现,提速 + 跨机。
- **vLLM / SGLang / Megatron**:明确排除。

---

## 12. 风险

- **重叠正确性**:权重版本错位(InferenceWorker 用旧权重采的数据被当新策略数据)、replay 读写竞争。
  缓解:weight 带 version、replay actor 串行化读写、parity 测试守住等价性。
- **抽取破坏单机路径**:若为复用而抽取 `OnlineCotrainRunner` 内联逻辑,可能动到单机行为。
  缓解:抽取即 parity 回归;非必要不抽取。
- **Ray 依赖引入**:`ray[default]>=2.47.0` 为**必装依赖**(用户要求);不做 import 隔离;
  单机路径默认不调 `ray.init()`(lazy 起集群),运行时不起集群、零开销。
- **单机"伪并行"**:单机多 GPU 下 actor 仍共享主机资源,重叠收益取决于 GPU 利用/调度;S5 验收
  需给出**可观测重叠证据**(time/ 命名空间下的 infer/learner 并发占比),否则 Ray 仅是复杂化。

---

## 附:与已合并单机 pipeline 的关系

`OnlineCotrainPipelineRunner`(已 merge)是本 backend 的**功能基线与 parity 对照**。Ray backend
不改它、不替它;两者共享 `OnlineReplay` / 更新步 / `_rollout_action` / `BaseRunner`
约定。用户在 config 选择走哪条。
