# 世界模型环境与运行器控制平面任务计划

> **给执行智能体的要求：** 实施本计划时使用 `superpowers:subagent-driven-development`
> 或 `superpowers:executing-plans`。每个任务使用 checkbox 跟踪，按顺序完成，不要跳过测试。

**目标：** 增加 RLinf 风格的世界模型环境路径，使
`RolloutWorker -> EnvWorker(real env)` 和 `RolloutWorker -> EnvWorker(world model env)`
共享同一个运行器控制平面，并在学习器更新后按采样边界同步世界模型 / 分类器权重快照。

**架构：** 运行器只做控制平面；工作器执行具体计算。`hidden` 是可选采样旁路字段；
`world_model + classifier` 打包成环境工作器下的世界模型环境后端；策略、世界模型、分类器
都有独立版本，想象轨迹必须记录这些版本。

**技术栈：** Python 3.11、Hydra、Ray 工作器组、PyTorch、pytest、现有
`OnlineCotrainRayRunner`、`LearnerWorker`、`EnvWorker`、`ReplayWorker` 和权重同步器。

---

## 范围

本计划不修改当前正在运行的训练任务。它只定义下一轮重构的实施任务。

目标拓扑：

```text
真实采样:
  RolloutWorker / PolicyWorker -> EnvWorker(real env) -> ReplayWorker / LearnerWorker

想象采样:
  RolloutWorker / PolicyWorker -> EnvWorker(WorldModelEnv) -> ReplayWorker / LearnerWorker

同步:
  LearnerWorker -> weight store -> RolloutWorker
  LearnerWorker -> weight store -> WorldModelEnv
```

## 文件结构

- 新增 `dreamervla/workers/inference/rollout_contract.py`
  - 定义统一采样输出契约。
  - 必需字段：`actions`。
  - 可选字段：`logprobs`、`values`、`policy_version`、`sidecars`。
- 修改 `dreamervla/workers/inference/rollout_inference_worker.py`
  - 把 `hidden` / `obs_embedding` 旁路字段改为显式可选。
- 修改 `dreamervla/workers/inference/inference_worker.py`
  - 返回同一类规范化契约，保留旧字典形状兼容。
- 新增 `dreamervla/envs/world_model/base_world_model_env.py`
  - 定义世界模型环境后端协议。
- 新增 `dreamervla/envs/world_model/latent_world_model_env.py`
  - 只做推理的环境后端，内部持有世界模型 / 分类器快照，暴露 `reset`、`step` 和可选 `chunk_step`。
- 修改 `dreamervla/workers/env/env_worker.py`
  - 保持对后端无感，支持世界模型后端的版本同步，不硬编码具体世界模型类。
- 修改 `dreamervla/workers/actor/learner_worker.py`
  - 固化策略、世界模型、分类器独立版本发布行为。
- 修改 `dreamervla/runners/online_cotrain_ray_runner.py`
  - 增加配置选择真实环境 / 世界模型环境的路径。
  - 在学习器更新边界触发快照同步。
  - 给轨迹标记版本。
- 新增 Hydra 配置
  - `configs/dreamervla/`
  - `configs/experiment/`
  - 提供一个小型合成世界模型环境冒烟测试路线。
- 新增单测
  - 采样契约测试
  - `hidden` 旁路字段默认关闭测试
  - 世界模型环境协议测试
  - 版本同步测试
  - 运行器连接测试

## 设计规则

- `hidden` 永远不是在线策略采样的必需输出。
- 世界模型环境内部包含分类器 / 成功验证器，用于奖励和结束标记推断。
- 世界模型 / 分类器同步发生在学习器更新后或显式间隔边界，不在每个环境步同步。
- 运行器拥有控制流和版本账本。
- 环境工作器只推进环境，不训练模型。
- 学习器是可训练权重的唯一发布源。

---

## 运行器具体控制逻辑

本节把运行器（`Runner`）的串行边界和可并行部分写清楚。后续任务实现时，以这里的控制流
为准。

### 控制状态

运行器持有控制状态，不持有业务计算逻辑：

```text
global_step
policy_version
wm_version
classifier_version
采样阶段状态
学习阶段状态
同步间隔
检查点间隔
指标命名空间
```

这些状态只用于编排、记录和校验。模型前向、环境推进、回放缓存和优化器更新分别放在对应
工作器中完成。

### 串行阶段

一轮在线训练的串行主线如下：

```text
1. 运行器解析并校验 Hydra 配置
2. 运行器按配置启动工作器组
3. 学习器构造训练副本
4. 推理工作器和世界模型环境加载初始推理快照
5. 运行器进入第 n 轮采样
6. 推理工作器根据观测批量产生动作
7. 环境工作器推进真实环境或世界模型环境
8. 轨迹写入回放缓存或直接进入学习队列
9. 数据达到更新条件后，学习器执行一次或一组更新
10. 学习器发布发生变化的组件快照
11. 运行器递增对应版本并在采样边界触发同步
12. 新采样轨迹记录 policy_version / wm_version / classifier_version
13. 运行器按间隔写日志、评测和检查点
14. 进入下一轮采样
```

必须串行等待的边界：

```text
配置校验完成后才能启动工作器
初始快照完成后才能开始采样
一条轨迹写入前必须带上使用过的版本号
学习器发布新快照后，运行器才能把对应版本标记为可同步
严格 on-policy 路径必须等当前采样批次完成后再更新
检查点写入必须使用一致的版本账本
```

### 可并行部分

工作器内部执行可以并行，但并行不能破坏版本语义：

```text
多个 EnvWorker 可以并行推进不同环境分片
RolloutWorker / PolicyWorker 可以把多个环境观测合成批量推理
ReplayWorker 可以在采样继续进行时接收轨迹写入
LearnerWorker 可以在满足更新条件后独立执行优化器更新
日志、视频和诊断可以异步写入，但不能反向决定训练维度
```

如果采用严格 on-policy 更新，采样和学习整体上是“采样一批 -> 学习一批 -> 同步 -> 再采样”。
如果采用异步联合训练，学习器可以和下一轮采样部分重叠，但必须满足：

```text
轨迹记录的是采样时实际使用的 policy_version / wm_version / classifier_version
学习器发布的新版本只影响同步边界之后的新轨迹
回放缓存按版本做陈旧度过滤
不允许每个环境步拉取新权重
```

### 真实环境路径

真实环境路径使用同一套运行器控制面：

```text
Runner
  -> RolloutWorker / PolicyWorker: obs -> action
  -> EnvWorker(real env): action -> next_obs, reward, done, info
  -> ReplayWorker / LearnerWorker: 写入真实轨迹
  -> LearnerWorker: 更新 policy / world_model / classifier 中被配置启用的组件
  -> Runner: 记录版本并在采样边界同步
```

真实环境奖励可以来自环境本身，也可以由成功验证器或奖励模型补充。补充奖励时，奖励模型仍然
作为配置声明的组件接入，不在运行器里写具体模型分支。

### 世界模型环境路径

世界模型环境路径不新增第二套控制面，只替换环境工作器内部后端：

```text
Runner
  -> RolloutWorker / PolicyWorker: obs -> action
  -> EnvWorker(world model env)
       -> WorldModelEnv: world_model 预测 next_obs
       -> WorldModelEnv: classifier / verifier 预测 reward 或 success
       -> WorldModelEnv: 生成 done 和 info
  -> ReplayWorker / LearnerWorker: 写入想象轨迹
  -> LearnerWorker: 更新 policy / world_model / classifier 中被配置启用的组件
  -> Runner: 记录版本并在采样边界同步
```

世界模型环境默认运行在 latent/token 空间，不要求生成图像。默认返回值保持最小：

```text
next_obs
reward
done
info
```

图像、`hidden`、token 旁路字段和诊断字段必须通过显式配置打开。

### 权重同步逻辑

同步只发生在学习器更新后或显式采样边界：

```text
LearnerWorker 完成 update
  -> 发布 policy / world_model / classifier 中发生变化的快照
  -> Runner 递增对应版本
  -> RolloutWorker 拉取 policy 快照
  -> WorldModelEnv 拉取 world_model + classifier 快照
  -> 后续新轨迹记录新版本
```

不允许把以下操作放进每个环境步：

```text
每步同步 policy
每步同步 world_model
每步单独远程调用 classifier
默认返回完整 hidden 旁路字段
```

### Ray 进程关系

Ray 路径下，运行器所在的 Python 进程仍然是控制平面主进程。它负责创建 Ray actor、发起
远程调用、等待必要结果和关闭 runtime。`ray::EnvWorker`、`ray::RolloutInferenceWorker`、
`ray::ReplayWorker`、`ray::LearnerWorker` 是执行平面 actor；它们不是独立训练入口。

因此，进程关系应理解为：

```text
python -m dreamervla.train
  -> Runner
     -> 创建 Ray actors
     -> 调用 actors
     -> 收集结果
     -> 编排同步、日志、评测、检查点
```

用户在系统进程里看到的非 Ray 主进程，就是运行器驱动进程。具体计算进程会以 Ray actor
形式出现。

### 延迟判断

把世界模型拆到世界模型环境中，不会天然造成大幅延迟。低延迟实现必须满足：

```text
世界模型环境常驻 world_model + classifier 推理快照
step/chunk_step 内部完成 next_obs/reward/done
只在采样边界同步权重
只返回最小 obs/reward/done/info
版本记录进轨迹
```

会造成明显延迟的做法是：

```text
策略工作器每步远程调用 world_model
world_model 每步远程调用 classifier
每步推送或拉取权重
默认返回完整 hidden / token 旁路字段
```

所以第一版实现应把世界模型作为环境工作器后端，而不是把世界模型前向拆成逐步远程服务。

---

## 任务 1：统一采样输出契约

**文件：**
- 新增：`dreamervla/workers/inference/rollout_contract.py`
- 测试：`tests/unit_tests/test_rollout_contract.py`

- [x] **步骤 1：写失败测试**

```python
import numpy as np

from dreamervla.workers.inference.rollout_contract import RolloutBatchOutput


def test_rollout_batch_output_requires_actions_only():
    actions = [np.zeros(7, dtype=np.float32)]
    out = RolloutBatchOutput(actions=actions)

    assert out.actions == actions
    assert out.logprobs is None
    assert out.values is None
    assert out.policy_version is None
    assert out.sidecars == {}


def test_rollout_batch_output_accepts_optional_hidden_sidecar():
    actions = [np.zeros(7, dtype=np.float32)]
    hidden = [np.ones(4, dtype=np.float16)]
    out = RolloutBatchOutput(actions=actions, sidecars={"hidden": hidden})

    assert out.sidecars["hidden"] == hidden
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_rollout_contract.py
```

预期：失败，原因是 `dreamervla.workers.inference.rollout_contract` 尚不存在。

- [x] **步骤 3：实现最小契约**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutBatchOutput:
    actions: list[Any]
    logprobs: list[Any] | None = None
    values: list[Any] | None = None
    policy_version: int | None = None
    sidecars: dict[str, list[Any]] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"actions": self.actions}
        if self.logprobs is not None:
            out["logprobs"] = self.logprobs
        if self.values is not None:
            out["values"] = self.values
        if self.policy_version is not None:
            out["policy_version"] = int(self.policy_version)
        out.update(self.sidecars)
        return out
```

- [x] **步骤 4：运行测试确认通过**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_rollout_contract.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/workers/inference/rollout_contract.py tests/unit_tests/test_rollout_contract.py
git commit -s -m "feat: add normalized rollout output contract"
```

---

## 任务 2：把 `RolloutInferenceWorker` 的 `hidden` 旁路字段改成显式可选

**文件：**
- 修改：`dreamervla/workers/inference/rollout_inference_worker.py`
- 测试：`tests/unit_tests/test_rollout_inference_worker.py`

- [x] **步骤 1：写失败测试**

新增测试：当配置 `emit_hidden_sidecar=False` 时，返回结果仍包含 `actions`，但不包含
`obs_embedding`。

```python
def test_rollout_inference_worker_can_disable_hidden_sidecar(monkeypatch):
    from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker

    class _Extractor:
        def prepare(self, obs, task_description):
            return {"obs": obs, "task_description": task_description}

    class _Bundle:
        def to(self, device):
            return self

        def make_extractor(self):
            return _Extractor()

        def predict_batch(self, preps):
            import numpy as np
            import torch

            return [(np.zeros((1, 7), dtype=np.float32), torch.ones(4)) for _ in preps]

    monkeypatch.setattr(
        "dreamervla.workers.inference.rollout_inference_worker._build_from_cfg",
        lambda cfg: _Bundle(),
    )

    cfg = {
        "decoder": {"target": "tests.fake.Bundle"},
        "action_dim": 7,
        "action_steps": 1,
        "emit_hidden_sidecar": False,
    }
    worker = RolloutInferenceWorker(cfg, {}, num_envs=1)
    worker.device = "cpu"
    worker.init()

    out = worker.forward_batch([{"task_description": "x"}], [0])
    assert "actions" in out
    assert "obs_embedding" not in out
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_rollout_inference_worker.py::test_rollout_inference_worker_can_disable_hidden_sidecar
```

预期：失败，因为 `RolloutInferenceWorker` 当前总是返回 `obs_embedding`。

- [x] **步骤 3：实现配置开关**

在 `RolloutInferenceWorker.__init__` 中加入：

```python
self._emit_hidden_sidecar = bool(self._cfg.get("emit_hidden_sidecar", True))
```

在 `forward_batch` 末尾返回：

```python
out: dict[str, list[Any]] = {"actions": actions}
if self._emit_hidden_sidecar:
    out["obs_embedding"] = hidden
return out
```

- [x] **步骤 4：运行测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_rollout_inference_worker.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/workers/inference/rollout_inference_worker.py tests/unit_tests/test_rollout_inference_worker.py
git commit -s -m "feat: make rollout hidden sidecar optional"
```

---

## 任务 3：定义世界模型环境后端协议

**文件：**
- 新增：`dreamervla/envs/world_model/base_world_model_env.py`
- 新增：`dreamervla/envs/world_model/__init__.py`
- 测试：`tests/unit_tests/test_world_model_env_contract.py`

- [x] **步骤 1：写失败测试**

```python
import numpy as np

from dreamervla.envs.world_model.base_world_model_env import WorldModelEnvProtocol


class _StubWorldEnv:
    wm_version = 3
    classifier_version = 4

    def reset(self, *, task_id=0, episode_id=0):
        return {"latent": np.zeros(2, dtype=np.float32)}, {"task_id": task_id}

    def step(self, action):
        return (
            {"latent": np.ones(2, dtype=np.float32)},
            1.0,
            True,
            False,
            {"wm_version": self.wm_version, "classifier_version": self.classifier_version},
        )

    def load_world_model_state(self, state_dict, version):
        self.wm_version = int(version)

    def load_classifier_state(self, state_dict, version):
        self.classifier_version = int(version)


def test_world_model_env_protocol_runtime_checkable():
    assert isinstance(_StubWorldEnv(), WorldModelEnvProtocol)
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_world_model_env_contract.py
```

预期：失败，因为模块尚不存在。

- [x] **步骤 3：实现协议**

```python
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorldModelEnvProtocol(Protocol):
    wm_version: int
    classifier_version: int

    def reset(self, *, task_id: int = 0, episode_id: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        ...

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        ...

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        ...
```

`dreamervla/envs/world_model/__init__.py`：

```python
from dreamervla.envs.world_model.base_world_model_env import WorldModelEnvProtocol

__all__ = ["WorldModelEnvProtocol"]
```

- [x] **步骤 4：运行测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_world_model_env_contract.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/envs/world_model tests/unit_tests/test_world_model_env_contract.py
git commit -s -m "feat: define world model env backend contract"
```

---

## 任务 4：增加合成版 `LatentWorldModelEnv`

**文件：**
- 新增：`dreamervla/envs/world_model/latent_world_model_env.py`
- 测试：`tests/unit_tests/test_latent_world_model_env.py`

- [x] **步骤 1：写失败测试**

```python
import numpy as np
import torch

from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv


class _TinyWM(torch.nn.Module):
    def forward(self, batch):
        latent = batch["latent"]
        action = batch["action"]
        return latent + action[..., : latent.shape[-1]]


class _TinyClassifier(torch.nn.Module):
    def forward(self, latent):
        return latent.sum(dim=-1, keepdim=True)


def test_latent_world_model_env_step_returns_env_tuple():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=7,
        success_threshold=0.5,
    )

    obs, info = env.reset(task_id=1, episode_id=2)
    assert "latent" in obs
    assert info["task_id"] == 1

    next_obs, reward, terminated, truncated, info = env.step(
        np.ones(7, dtype=np.float32)
    )

    assert "latent" in next_obs
    assert reward > 0.0
    assert terminated is True
    assert truncated is False
    assert info["wm_version"] == 0
    assert info["classifier_version"] == 0
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_latent_world_model_env.py
```

预期：失败，因为 `latent_world_model_env.py` 尚不存在。

- [x] **步骤 3：实现合成环境**

实现一个只做推理的环境，行为如下：

```text
保存当前 latent
执行 world_model({"latent": latent, "action": action})
使用 classifier(next_latent) 得到奖励分数
terminated = score >= success_threshold
truncated = elapsed_steps >= max_episode_steps
info 里写入 wm_version 和 classifier_version
```

该文件不要导入具体 DreamerVLA 世界模型类。构造函数接收已经构造好的 module，或接收 Hydra
构造出的 module。

- [x] **步骤 4：运行测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_latent_world_model_env.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/envs/world_model/latent_world_model_env.py tests/unit_tests/test_latent_world_model_env.py
git commit -s -m "feat: add latent world model env backend"
```

---

## 任务 5：通过环境工作器同步世界模型和分类器快照

**文件：**
- 修改：`dreamervla/workers/env/env_worker.py`
- 测试：`tests/unit_tests/test_env_worker_world_model_sync.py`

- [x] **步骤 1：写失败测试**

创建一个桩环境，它暴露 `load_world_model_state` 和 `load_classifier_state`。
调用环境工作器新方法后，断言版本更新。

```python
def test_env_worker_forwards_world_model_and_classifier_sync():
    from dreamervla.workers.env.env_worker import EnvWorker

    class _Env:
        def __init__(self):
            self.wm_version = 0
            self.classifier_version = 0

        def load_world_model_state(self, state_dict, version):
            self.wm_version = int(version)

        def load_classifier_state(self, state_dict, version):
            self.classifier_version = int(version)

    worker = EnvWorker(env_cfg={"target": "unused"}, task_id=0, replay=None)
    worker.env = _Env()

    worker.load_world_model_state({}, version=5)
    worker.load_classifier_state({}, version=7)

    assert worker.env.wm_version == 5
    assert worker.env.classifier_version == 7
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_env_worker_world_model_sync.py
```

预期：失败，因为环境工作器还没有这些转发方法。

- [x] **步骤 3：实现转发方法**

在 `EnvWorker` 中增加：

```python
def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
    env = self._active_env()
    if not hasattr(env, "load_world_model_state"):
        raise RuntimeError("active env does not support world model state sync")
    env.load_world_model_state(state_dict, int(version))


def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
    env = self._active_env()
    if not hasattr(env, "load_classifier_state"):
        raise RuntimeError("active env does not support classifier state sync")
    env.load_classifier_state(state_dict, int(version))
```

同时增加 `_active_env()`。进程内环境直接返回 `self.env`。如果是子进程环境，需要通过 pipe
命令路由，不能直接访问子进程里的环境对象。

- [x] **步骤 4：运行测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_env_worker_world_model_sync.py tests/unit_tests/test_env_worker_spawn_recovery.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/workers/env/env_worker.py tests/unit_tests/test_env_worker_world_model_sync.py
git commit -s -m "feat: sync world model env snapshots through env worker"
```

---

## 任务 6：固化学习器的独立组件版本发布行为

**文件：**
- 修改：`dreamervla/workers/actor/learner_worker.py`
- 测试：`tests/unit_tests/test_learner_worker_component_versions.py`

- [x] **步骤 1：写表征测试**

使用小型伪组件，断言以下调用会把不同组件以独立版本推送到同步器：

```text
sync_weights("world_model", 3)
sync_weights("classifier", 4)
sync_weights("policy", 5)
```

- [x] **步骤 2：运行测试确认当前行为**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_learner_worker_component_versions.py
```

预期：通过。当前 `LearnerWorker.sync_weights` 已经接受 `self.components` 中存在的任意组件；
这个任务主要是把行为锁成回归测试。

- [x] **步骤 3：仅在表征测试失败时修改代码**

预期路径是不改生产代码。如果测试失败，说明 `sync_weights` 只处理了 `policy`，则扩展为接受：

```text
policy
world_model
classifier
critic
```

不要针对具体模型类写分支。

- [x] **步骤 4：运行测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_learner_worker_component_versions.py tests/unit_tests/test_learner_worker_manual_precision.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add dreamervla/workers/actor/learner_worker.py tests/unit_tests/test_learner_worker_component_versions.py
git commit -s -m "feat: version learner component weight sync"
```

---

## 任务 7：在运行器中加入世界模型环境同步控制

**文件：**
- 修改：`dreamervla/runners/online_cotrain_ray_runner.py`
- 测试：`tests/unit_tests/test_online_cotrain_ray_runner.py`

- [x] **步骤 1：写失败测试**

增加一个运行器层面的单测，使用伪工作器组验证串行同步边界：

```text
学习器更新后:
  policy 更新时 policy_version 增加
  world_model 更新时 wm_version 增加
  classifier 更新时 classifier_version 增加
  推理工作器收到 policy 拉取请求
  支持世界模型的环境工作器收到 WM/classifier 同步请求
```

测试还必须断言：

```text
采样中途不触发权重同步
轨迹写入时包含采样实际使用的版本号
下一轮采样才使用学习器刚发布的新版本
```

建议测试骨架：

```python
def test_runner_syncs_snapshots_only_at_rollout_boundary():
    runner = _make_runner_with_fake_worker_groups()

    runner._policy_version = 0
    runner._wm_version = 0
    runner._classifier_version = 0

    runner._begin_rollout_round()
    runner._record_transition({"obs": 1, "action": 2})

    assert runner._fake_policy_worker.pull_calls == []
    assert runner._fake_env_worker.wm_sync_calls == []
    assert runner._fake_replay.last_transition["policy_version"] == 0
    assert runner._fake_replay.last_transition["wm_version"] == 0
    assert runner._fake_replay.last_transition["classifier_version"] == 0

    runner._mark_learner_update_result(
        {
            "policy": {"updated": True},
            "world_model": {"updated": True},
            "classifier": {"updated": True},
        }
    )
    runner._sync_after_rollout_boundary()

    assert runner._policy_version == 1
    assert runner._wm_version == 1
    assert runner._classifier_version == 1
    assert runner._fake_policy_worker.pull_calls == [1]
    assert runner._fake_env_worker.wm_sync_calls == [1]
    assert runner._fake_env_worker.classifier_sync_calls == [1]
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py -k world_model_env_sync
```

预期：失败，因为运行器当前不会把世界模型 / 分类器快照同步到环境工作器。

- [x] **步骤 3：写并行调度表征测试**

增加一个不启动真实 Ray 的表征测试，验证运行器把可并行工作交给工作器组，而不是在运行器里
串行执行模型前向或环境推进：

```python
def test_runner_dispatches_rollout_work_to_worker_groups():
    runner = _make_runner_with_fake_worker_groups(num_env_workers=3)

    runner._dispatch_rollout_round()

    assert runner._fake_policy_worker.forward_batch_calls == 1
    assert runner._fake_env_group.step_calls_by_worker == [1, 1, 1]
    assert runner._fake_learner.optimizer_step_calls == 0
```

这个测试只检查调度边界，不要求真实并发。真实 Ray 并发由 e2e 冒烟测试覆盖。

- [x] **步骤 4：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py -k "world_model_env_sync or dispatches_rollout"
```

预期：失败，因为当前运行器缺少世界模型环境同步路径，或缺少可测试的调度边界方法。

- [x] **步骤 5：实现同步边界**

增加控制状态：

```python
policy_version = 0
wm_version = 0
classifier_version = 0
```

学习器更新后：

```text
policy 更新 -> learner.sync_weights("policy", policy_version)
WM 更新 -> learner.sync_weights("world_model", wm_version)
classifier 更新 -> learner.sync_weights("classifier", classifier_version)
```

采样边界：

```text
PolicyWorker 拉取 policy
支持 WorldModelEnv 的 EnvWorker 拉取或加载 world_model 与 classifier
```

transition 或 episode 必须记录：

```text
policy_version
wm_version
classifier_version
```

- [x] **步骤 6：实现调度边界**

运行器新增或整理出可测试的内部边界方法。方法名可以按现有代码风格调整，但语义必须一致：

```text
_begin_rollout_round()
_dispatch_rollout_round()
_record_transition(...)
_mark_learner_update_result(...)
_sync_after_rollout_boundary()
```

实现要求：

```text
_dispatch_rollout_round 只调用推理工作器和环境工作器，不调用 optimizer.step
_record_transition 写入采样时的版本号
_mark_learner_update_result 只记录哪些组件更新，不直接同步远端工作器
_sync_after_rollout_boundary 统一执行 policy / world_model / classifier 同步
```

- [x] **步骤 7：运行运行器测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py tests/unit_tests/test_online_replay_staleness.py
```

预期：通过。

- [ ] **步骤 8：提交**

```bash
git add dreamervla/runners/online_cotrain_ray_runner.py tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "feat: sync world model env snapshots from runner"
```

---

## 任务 8：增加真实环境与世界模型环境的 Hydra 路由

**文件：**
- 新增：`configs/dreamervla/ray_online_cotrain_world_model_env_tiny.yaml`
- 新增：`configs/experiment/online_cotrain_ray_world_model_env_tiny.yaml`
- 测试：`tests/unit_tests/test_world_model_env_config.py`

- [x] **步骤 1：写配置 compose 测试**

```python
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_world_model_env_tiny_experiment_composes():
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_world_model_env_tiny",
                "logger=tensorboard",
            ],
        )
    assert cfg.runner_name == "online_cotrain_ray"
    target = str(cfg.env.get("target", cfg.env.get("_target_", "")))
    assert "world_model" in target
```

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_world_model_env_config.py
```

预期：失败，因为配置还不存在。

- [x] **步骤 3：新增配置**

新增小型合成配置，要求：

```text
使用 OnlineCotrainRayRunner
使用 EnvWorker + LatentWorldModelEnv 后端
使用小型 policy/world_model/classifier 测试模块
默认关闭采集阶段的 hidden 旁路字段
sync interval = 1
所有维度都在 Hydra 中显式设置
```

不要从代码默认值推断旁路特征维度。

- [x] **步骤 4：运行配置测试**

运行：

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_world_model_env_config.py
```

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add configs/dreamervla/ray_online_cotrain_world_model_env_tiny.yaml configs/experiment/online_cotrain_ray_world_model_env_tiny.yaml tests/unit_tests/test_world_model_env_config.py
git commit -s -m "feat: add world model env ray cotrain route"
```

---

## 任务 9：端到端冒烟测试与文档更新

**文件：**
- 修改：`docs/ray_online_cotrain_backend.md`
- 修改：`docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`
- 测试：`tests/e2e_tests/test_world_model_env_ray_smoke.py`

- [x] **步骤 1：增加带开关的端到端冒烟测试**

该测试只在设置 `DVLA_WORLD_MODEL_ENV_SMOKE=1` 时运行。

冒烟测试命令：

```bash
PYTHONPATH=. WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
python -m dreamervla.train \
  experiment=online_cotrain_ray_world_model_env_tiny \
  logger=tensorboard \
  training.out_dir=/tmp/dvla_world_model_env_smoke \
  rollout.steps=9
```

断言：

```text
resolved_config.yaml 存在
run_manifest.json 存在
metrics 包含 sync/policy_version
metrics 包含 sync/wm_version
metrics 包含 sync/classifier_version
采样不依赖 hidden 旁路字段也能完成
```

- [x] **步骤 2：运行非 gated 单测**

运行：

```bash
PYTHONPATH=. pytest -q \
  tests/unit_tests/test_rollout_contract.py \
  tests/unit_tests/test_rollout_inference_worker.py \
  tests/unit_tests/test_world_model_env_contract.py \
  tests/unit_tests/test_latent_world_model_env.py \
  tests/unit_tests/test_env_worker_world_model_sync.py \
  tests/unit_tests/test_online_cotrain_ray_runner.py \
  tests/unit_tests/test_world_model_env_config.py
```

预期：通过。

- [x] **步骤 3：更新说明文档**

文档必须说明：

```text
Runner 是控制平面
PolicyWorker 的 hidden 旁路字段是可选项
WorldModelEnv 是 EnvWorker 后端
classifier/verifier 放在 WorldModelEnv 推理快照内部
LearnerWorker 发布 policy/WM/classifier 版本
同步发生在采样边界
```

- [x] **步骤 4：资源可用时运行带开关的冒烟测试**

运行：

```bash
DVLA_WORLD_MODEL_ENV_SMOKE=1 PYTHONPATH=. pytest -q tests/e2e_tests/test_world_model_env_ray_smoke.py
```

预期：在 Ray 可用机器上通过。

- [ ] **步骤 5：提交**

```bash
git add docs/ray_online_cotrain_backend.md docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md tests/e2e_tests/test_world_model_env_ray_smoke.py
git commit -s -m "docs: explain world model env runner topology"
```

---

## 验证门槛

标记计划完成前，至少运行：

```bash
PYTHONPATH=. pytest -q \
  tests/unit_tests/test_rollout_contract.py \
  tests/unit_tests/test_rollout_inference_worker.py \
  tests/unit_tests/test_world_model_env_contract.py \
  tests/unit_tests/test_latent_world_model_env.py \
  tests/unit_tests/test_env_worker_world_model_sync.py \
  tests/unit_tests/test_learner_worker_component_versions.py \
  tests/unit_tests/test_online_cotrain_ray_runner.py \
  tests/unit_tests/test_world_model_env_config.py
```

完整 GPU/Ray 运行通过前，不能宣称生产可用。

## 默认决策

- 第一版世界模型环境默认返回 latent/token observation。图像渲染只作为诊断选项，
  通过显式配置打开。
- 现有检查点和配置中的可训练组件名称保持 `classifier`。协议层角色叫
  `SuccessVerifier`，用文档和类型名说明职责，不重命名检查点键。
- `policy_version` 在 actor/policy 更新后递增。`wm_version` 在 world-model 更新后递增。
  `classifier_version` 在 classifier/verifier 更新后递增。某个阶段没有更新对应组件，
  就不递增对应版本。
- 第一版想象轨迹进入 `ReplayWorker`，复用现有陈旧度过滤和版本记账。只有回放开销被测量为
  瓶颈后，才考虑新增直接轨迹通道路径。
