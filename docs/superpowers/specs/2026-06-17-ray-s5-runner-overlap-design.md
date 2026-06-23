# 设计(S5):OnlineCotrainRayRunner + 重叠调度 —— 串链、config、AGENTS.md 软化

- 日期:2026-06-17
- 状态:子项目 spec,待 review(第 5/5 份)
- 主题:落地总览 §7 的 **S5**:把 S1–S4 串成一个可选 backend runner `OnlineCotrainRayRunner`,做
  **基础 infer→step→learn 重叠**;config 选择 backend;`validate_cfg` 扩展;**AGENTS.md 软化**。
- 范围:**仅在线 cotrain loop**(决策:离线 warmup 仍走单机;此处用 warmup ckpt 作 init)。
- 关联:
  - S1–S4 全部产物(scheduler / EnvWorker / ReplayWorker / InferenceWorker / LearnerWorker / WeightSyncer)
  - parity 基线:`dreamervla/runners/online_cotrain_pipeline_runner.py`(`OnlineCotrainPipelineRunner`)
  - `dreamervla/runners/base_runner.py`(`BaseRunner`:artifacts/checkpoint/logging)
  - `dreamervla/runners/__init__.py`(`PUBLIC_RUNNERS` 注册)、`dreamervla/config.py`(`validate_cfg`)
  - `configs/`(新增 `scheduler/` 组)、`AGENTS.md:6`/`:69-71`、`CLAUDE.md`(RLinf Alignment Snapshot)
- 参考实现(RLinf):`RLinf/rlinf/runners/embodied_runner.py`(`update_rollout_weights` + 并发 env/rollout + actor 训练)

---

## 1. 范围与边界

- **范围内**:`runners/online_cotrain_ray_runner.py`(`OnlineCotrainRayRunner`)、`configs/scheduler/`
  组、`validate_cfg` 扩展、AGENTS.md/CLAUDE.md 软化、端到端 smoke + 与单机 parity。
- **范围外**:离线 warmup 的 Ray 化、全双缓冲预取、多节点、auto-scaling(§7 TODO)。

---

## 2. 目标 / 非目标

**目标**
1. 一个 opt-in backend runner,把 S1–S4 串成 infer→step→learn 在线循环,与单机
   `OnlineCotrainPipelineRunner` **行为等价**(同 config/seed 下指标对齐到容差),但 env/infer/learner
   分置异构 actor 并 **基础重叠**。
2. **复用 `BaseRunner`**:run-artifact 布局、checkpoint(`world_model`/`policy`/`classifier`/`critic`)、
   多后端日志、`train/ eval/ env/ rollout/ time/` 命名空间。
3. **config 选 backend**:`runner_name="online_cotrain_ray"`,经 `experiment=<name>` 选;**不新增顶层
   route YAML**;新增 `configs/scheduler/` 组。`validate_cfg` 早校验 GPU/worker/同步频率。
4. **warmup 复用**:离线 warmup 仍单机产出 `wm_warmup.ckpt`/`classifier_warmup.ckpt`;Ray runner 经
   `init.world_model_state_ckpt`/`init.classifier_state_ckpt` 加载作初值。
5. **lazy 起集群**:仅本 runner(选到 Ray)才 import `dreamervla.scheduler` 并 `Cluster()`;单机路径不碰。
6. 给出**可观测重叠证据**(`time/` 下 infer/learner 并发占比),否则 Ray 仅是复杂化。

**非目标**
- 不替换、不改单机 `OnlineCotrainPipelineRunner`(它是 parity 基线)。
- 不在 Ray runner 内做离线 warmup(§7 TODO)。

---

## 3. 模块与控制流

### `runners/online_cotrain_ray_runner.py`

```python
class OnlineCotrainRayRunner(BaseRunner):
    runner_name = "online_cotrain_ray"
    runner_status = "current"; runner_family = "actor"

    def _build_components(self):
        from dreamervla.scheduler import Cluster, WorkerGroup, Channel, \
            PackedPlacementStrategy, NodePlacementStrategy        # lazy import(只此路径)
        cluster = Cluster(cfg.scheduler)
        replay = ReplayWorker.create_group(...).launch(cluster, NodePlacementStrategy(1))
        envs   = WorkerGroup(EnvWorker, ..., replay=replay).launch(cluster, NodePlacementStrategy(k))
        infer  = WorkerGroup(InferenceWorker, ...).launch(cluster, PackedPlacementStrategy(infer_gpu, infer_gpu))
        learner= WorkerGroup(LearnerWorker, ..., replay=replay, syncer=syncer)\
                   .launch(cluster, PackedPlacementStrategy(learner_gpu, learner_gpu))
        # Channels: obs / action

    def run(self):
        # 预热:envs 各 reset;循环:
        for step in range(total):
            rollout_h = self._rollout_once()          # gather obs->infer->scatter action->env.step->replay
            learn_h   = self._learn_once(step)        # learner.update(wm/cls/rl) + sync_weights
            wait([rollout_h, learn_h])                # 基础重叠:两者并发,单缓冲 barrier
            if step % weight_sync_every == 0: infer.update_weights(...)  # pull 最新 wm/policy
            self._maybe_eval_checkpoint_log(step)     # 复用 BaseRunner
```
- **相位切换**:按 `training.warmup_steps` 切 warmup(只 WM+CLS)/ cotrain(加 RL),复用单机语义。
- **重叠(基础)**:rollout 与 learner 各自 actor 并发(不同 GPU),`run()` 用异步 `ObjectRef` 同发、
  到 barrier 收;单缓冲。可观测重叠 = 记录两者 wall-clock 重合度到 `time/`。

### `configs/scheduler/`(新组)
`num_env_workers`、`infer_gpu`、`learner_gpu`、`weight_sync_every`、`channel_maxsize`、`placement` 名。

### `validate_cfg` 扩展
GPU 数 ≥ (infer 1 + learner 1);`infer_gpu`≠`learner_gpu`;`num_env_workers`≥1;`weight_sync_every`≥1 且
与 horizon/chunk 一致;选 Ray backend 时 `ray` 可 import。

---

## 4. AGENTS.md / CLAUDE.md 软化(并入本子项目)

- `AGENTS.md:6`、`:69-71`:由"无 Ray / 不引入 Ray stack" 改为
  **"单机 torchrun 为默认主线;Ray 作为 opt-in distributed backend 可用,但不得成为默认、不得侵入
  单机路径(单机运行时不起 ray 集群)。"**
- `CLAUDE.md` RLinf Alignment Snapshot 同步:Ray 由"明确排除"改为"opt-in backend,默认仍单机"。

---

## 5. 测试 / 验收(真 ray + 真负载;S5 是真实测试的汇聚点)

`tests/e2e_tests/test_s5_ray_cotrain_smoke.py`(GPU,标记 slow):
1. **端到端 smoke**:tiny config(1–2 EnvWorker、几十步、小模型/warmup ckpt)→ Ray runner 跑完不崩,
   产出 run-artifact(checkpoints/log/`resolved_config.yaml`/`run_manifest.json`)。
2. **与单机 parity/等价**:同 seed/同 config 下,Ray runner 与 `OnlineCotrainPipelineRunner` 的关键指标
   (WM/CLS/RL loss、env 成功率)对齐到容差内。
3. **可观测重叠**:`time/` 指标显示 infer 与 learner wall-clock 有重合(重叠比 > 0),并随负载变化。
4. **config 早校验**:GPU 不足 / `infer_gpu==learner_gpu` / 缺字段时 `validate_cfg` 早报错。
5. **单机零侵入**:跑单机 runner 时**不**起 ray 集群(无 `ray.init`)、不 import `dreamervla.scheduler`。

**验收**:1–5 全过(2、3 需 GPU,heavy);`ruff`/类型检查过;单机回归绿。

---

## 6. 实现顺序(TDD)

1. `_build_components`(起 cluster + 4 类 worker + channel)+ smoke 测试 1。
2. `run` 的 rollout/learn 串链(先串行)→ 与单机 parity 测试 2。
3. 基础重叠(并发 + barrier + 重叠度记录)→ 测试 3。
4. `configs/scheduler/` + `validate_cfg` 扩展 → 测试 4。
5. lazy 起集群 + 单机零侵入 → 测试 5。
6. AGENTS.md/CLAUDE.md 软化 + `PUBLIC_RUNNERS` 注册。

---

## 7. TODO(更激进版本 / 后续,本期不做)

- **全双缓冲预取重叠**:边训 learner 边预取下一批 rollout,双缓冲最大化掩盖互等(替换基础单缓冲 barrier);
  需 channel `async_op`(S1 §7)+ 权重版本/读写竞争的更强守护。
- **完整流水线(含离线 warmup)Ray 化**:Ray runner 复刻 `OnlineCotrainPipelineRunner` 全程(离线
  warmup 预灌注 + 在线 cotrain),与单机完全对等。
- **多节点**:head/worker 跨机、sharded `ReplayWorker` 服务、跨机 NCCL learner(DDP/FSDP)+ 权重同步。
- **dynamic scheduler / auto-scaling**:仿 RLinf `ComponentManager`/`RolloutScalingScheduler`,按队列深度
  自动扩缩 env/infer 副本数。

---

## 8. 风险

- **等价性**:重叠/异步引入的权重版本错位、replay 读写竞争可能让 Ray 训练偏离单机 → 测试 2 等价性 +
  S4 `version` 守;先串行通过 parity,再开重叠。
- **重叠收益不显**:单机多 GPU 下 actor 共享主机,重叠收益取决于 GPU 利用 → 测试 3 必须给正的重叠比,
  否则回退评估(Ray 是否值得)。
- **抽取破坏单机**:若为复用而抽取 `OnlineCotrainRunner` 内联逻辑,可能动到单机行为 → 抽取即 parity 回归。
- **config 漂移**:新增 `scheduler/` 组与现有组耦合 → `validate_cfg` 早校验 + smoke 守。
