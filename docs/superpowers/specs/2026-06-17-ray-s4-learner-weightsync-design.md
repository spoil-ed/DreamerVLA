# 设计(S4):LearnerWorker + WeightSyncer —— 训练侧 Ray 化 + 权重回灌

- 日期:2026-06-17
- 状态:子项目 spec,待 review(第 4/5 份)
- 主题:落地总览 §7 的 **S4**:把 WM/classifier/RL 更新步做成一个 GPU actor `LearnerWorker`(从
  `ReplayWorker` 采样、跑相位更新),并实现 `WeightSyncer` 把更新后的 `world_model`+`policy` 权重
  经 **object-store** 回灌给 `InferenceWorker`。
- 范围:**仅 S4**。单 GPU learner(DDP 后置)、object-store 权重同步(NCCL 后置)。串链/重叠是 S5。
- 关联:
  - S1 骨架、S2 `ReplayWorker.sample`、S3 `InferenceWorker.update_weights`
  - 相位与更新步(`online_cotrain_runner.py:400-477`):
    - WM:`world_model_pretrain_step`(`dreamervla/algorithms/dreamervla.py:62`)
    - CLS:`online_classifier_update_step`(`dreamervla/runners/online_dreamervla.py:394`)
    - RL:`dino_wmpo_outcome_step`(`dreamervla/algorithms/ppo/outcome.py:138`,只训 `policy`)
  - 模块:`world_model`/`classifier`/`policy`/`critic` + 各 optimizer
- 参考实现(RLinf):`RLinf/rlinf/workers/actor/`、`RLinf/rlinf/hybrid_engines/weight_syncer/`

---

## 1. 范围与边界

- **范围内**:`workers/actor/learner_worker.py`(`LearnerWorker`)、`hybrid_engines/weight_syncer/`
  (`base.py` ABC + `objectstore.py`);相位更新 + 权重 push。
- **范围外**:多 GPU DDP learner(§7 TODO)、NCCL 权重同步(§7 TODO)、runner/重叠(S5)。

---

## 2. 目标 / 非目标

**目标**
1. `LearnerWorker` 从 `ReplayWorker` 采样,按现有相位顺序跑更新步,**复用** `world_model_pretrain_step`/
   `online_classifier_update_step`/`dino_wmpo_outcome_step`,行为对齐单机。
2. **相位/冻结对齐**(据勘探):
   - WM 相位:训 `world_model`;`policy`/`critic`/`classifier` eval。
   - CLS 相位:训 `classifier`。
   - RL 相位(cotrain):训 `policy`;`world_model`/`classifier` 冻结(断言守)。
3. `WeightSyncer`(object-store):**每轮同步 `world_model`(WM 相位后)+ `policy`(RL 相位后)**;带
   `version` 防陈旧;`InferenceWorker` pull 后换权。
4. 单 GPU learner;接口为多 GPU DDP 预留(§7)。

**非目标**
- 不同步 `critic`/`classifier`(rollout 推理不用,据勘探);`encoder` 冻结不传。
- 不改更新步函数本身(只做 actor 封装与调用编排)。

---

## 3. 模块与 API

### 3.1 `workers/actor/learner_worker.py`

```python
class LearnerWorker(Worker):                     # GPU(默认 1 块)actor
    def __init__(self, model_cfg, init_ckpt, train_cfg,
                 replay: "ray.ActorHandle", syncer: "WeightSyncer") -> None: ...
    def init(self) -> None: ...
        # 建 world_model/classifier/policy/critic + optimizers(同单机 runner 装配)
    def update(self, phase: str, num_steps: int) -> dict[str, float]:
        # phase in {"wm","cls","rl"};内部 batch = ray.get(replay.sample.remote(bsz));
        #   wm  -> world_model_pretrain_step(...)        训 world_model
        #   cls -> online_classifier_update_step(...)    训 classifier
        #   rl  -> dino_wmpo_outcome_step(...)           训 policy(wm/cls 冻结)
        # 返回 train/ 指标
    def sync_weights(self, what: str, version: int) -> None:
        # what=="wm"  -> syncer.push(world_model.state_dict(), version)
        # what=="policy" -> syncer.push(policy.state_dict(), version)
    def state_dicts(self) -> dict: ...           # checkpoint 用(world_model/policy/critic/classifier)
```
- learner 持 `ReplayWorker` 句柄(采样)与 `WeightSyncer`(push);相位编排由 S5 runner 驱动
  (本期测试直接调 `update`/`sync_weights`)。

### 3.2 `hybrid_engines/weight_syncer/`

```python
# base.py
class WeightSyncer(ABC):
    def push(self, state_dict: dict, version: int) -> None: ...
    def pull(self, model: nn.Module, version: int) -> bool: ...   # 载入最新;无更新返回 False

# objectstore.py
class ObjectStoreWeightSyncer(WeightSyncer):
    # push: cpu_sd = {k: v.cpu() for ...}; ref = ray.put(cpu_sd);
    #       _WeightStore.set.remote(key, version, ref)            # 具名 actor 持 {key:(version,ref)}
    # pull: (ver, ref) = ray.get(_WeightStore.get.remote(key));
    #       if ver > local_ver: model.load_state_dict(ray.get(ref)); return True
```
- 单机 object store = plasma 共享内存,`ray.put`/`ray.get` 廉价;`version` 单调,pull 只在更新时换权。
- `key` 区分 `"world_model"`/`"policy"`。

---

## 4. 数据流要点

```
LearnerWorker:                                   InferenceWorker:
  batch = replay.sample(bsz)                       每 weight_sync_every 步:
  m = update("wm"); sync_weights("wm", v)            wm 变了 -> pull world_model
  update("cls")                                       policy 变了 -> pull policy
  m = update("rl"); sync_weights("policy", v)
```
- 相位顺序、warmup vs cotrain 的 RL 开关 = 复用单机语义(S5 runner 按 `warmup_steps` 切)。

---

## 5. 测试 / 验收(真 ray + 真模型;小规模)

`tests/e2e_tests/test_s4_learner_weightsync.py`:
1. **权重同步正确性**(轻、不依赖大模型):`ObjectStoreWeightSyncer` push 一个小 `state_dict`(v=1)→
   另一进程/模型 `pull` → `state_dict` `allclose` 且 `version` 前进;重复 pull 同版本返回 `False`(不换)。
2. **相位更新可跑**:`LearnerWorker` + `ReplayWorker`(预灌合成 episode)→ `update("wm")`/`update("cls")`/
   `update("rl")` 各跑数步,loss 有限、可下降;相位内**冻结断言**成立(RL 相位 `world_model`/`classifier`
   不更新)。
3. **端到端 sync**:learner 训 1 步 → `sync_weights("policy")` → `InferenceWorker.update_weights`/`pull`
   后,policy 权重等于 learner 侧(`allclose`)。

**验收**:1 必过(轻);2、3 需 GPU + 模型(heavy,标记 slow);`ruff`/类型检查过。

---

## 6. 实现顺序(TDD)

1. `weight_syncer/base.py` + `objectstore.py` + `_WeightStore` actor → 测试 1。
2. `LearnerWorker.init`/`update`(相位封装,复用更新步)→ 测试 2。
3. `sync_weights` 接 `InferenceWorker.update_weights` → 测试 3。

---

## 7. TODO(更激进版本 / 后续,本期不做)

- **NCCL broadcast WeightSyncer**:`WeightSyncer` 第二实现,learner GPU 直连 InferenceWorker GPU 广播
  权重(不过 CPU/序列化),提速 + 跨机;需 S1 预留的 collective group。
- **多 GPU DDP/FSDP learner**:`LearnerWorker` 跨多卡,actor 内起 `torch.distributed` 进程组;
  placement 用 `PackedPlacementStrategy(1, N)`。
- **bucketed / async 权重同步**:仿 RLinf `BucketWeightSyncer`/`PatchWeightSyncer`,分桶 + 通信重叠。
- **同步 critic/classifier**:若将来 rollout 推理用到(当前不用),扩 sync 集合。

---

## 8. 风险

- **权重版本错位**:learner push 与 infer pull 跨步并发(S5 重叠)→ 用 `version` + 原子换权;parity/
  等价性测试守住(S5)。
- **冻结泄漏**:相位间 train/eval 切换若漏,导致该冻结的被更新 → 测试 2 的冻结断言守。
- **object store 体积**:大 `state_dict` 频繁 `ray.put` 占共享内存;`weight_sync_every` 控频(S5 config)。
- **采样阻塞**:`replay.sample` 与 env 写入竞争单 `ReplayWorker`;单机够用,瓶颈留 S2 §7 分片。
