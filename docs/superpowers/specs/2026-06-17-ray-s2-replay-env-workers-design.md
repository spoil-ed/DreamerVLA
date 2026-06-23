# 设计(S2):ReplayWorker + EnvWorker —— 采集侧 Ray 化(仿 RLinf workers/env)

- 日期:2026-06-17
- 状态:子项目 spec,待 review(第 2/5 份)
- 主题:落地总览 §7 的 **S2**:把 env 步进做成**每 env 一个 Ray actor**(`EnvWorker`),把
  `OnlineReplay` 包成**共享 Ray actor**(`ReplayWorker`);EnvWorker 采到的 episode 在 `done` 时
  push 给 ReplayWorker。**复用现成 env 契约与 `OnlineReplay`,不改其逻辑。**
- 范围:**仅 S2**。在 S1 骨架上搭这两个 worker + 采集侧数据流(env→replay)。InferenceWorker(S3)
  尚不存在,本期用 dummy 动作源驱动 EnvWorker。
- 关联:
  - S1 骨架:`docs/superpowers/specs/2026-06-17-ray-s1-scaffolding-design.md`(`Worker`/`WorkerGroup`)
  - `dreamervla/runners/online_replay.py`(`OnlineReplay`:`add_episode`/`sample`)
  - `dreamervla/runners/vec_rollout_env.py`、`dreamervla/envs/train_env.py`(env 契约 + `make_transition`)
  - producer loop:`dreamervla/runners/online_cotrain_runner.py:353-386`
- 参考实现(RLinf):`RLinf/rlinf/workers/env/env_worker.py`(env worker 高层 rollout 循环)

---

## 1. 范围与边界

- **范围内**:`workers/env/env_worker.py`(`EnvWorker`)、`workers/replay/replay_worker.py`
  (`ReplayWorker`);采集侧数据流:EnvWorker 步进 → 攒 episode → `done` push ReplayWorker。
- **范围外**:推理(S3,本期用 dummy 动作 + dummy `obs_embedding` 驱动)、learner/采样消费(S4)、
  重叠调度/runner(S5)、LIBERO 真负载端到端(S5)。

---

## 2. 目标 / 非目标

**目标**
1. `EnvWorker` 单 env 步进 + auto-reset,产出与现有**逐字段一致**的 transition(§4),`done` 时把整条
   episode 交给 `ReplayWorker`。
2. `ReplayWorker` 把 `OnlineReplay` 暴露为 actor(`add_episode`/`sample`/`size`),**构造参数透传**、
   容量/淘汰/task 平衡沿用其现有策略。
3. **保留 per-episode 身份元数据**(`task_id`/`episode_success` 等由 `OnlineReplay.add_episode` 计算,
   见 §4);EnvWorker 只需保证每 step dict 字段齐、首末步标志正确。

**非目标**
- 不改 `OnlineReplay`、不改 env 契约 / `make_transition`。
- 不做分片/多节点 replay(§7 TODO)。
- 不在 EnvWorker 内嵌 `SubprocVecEnv`(actor 本身即进程隔离;每 worker 持 1 真 env)。

---

## 3. 模块与 API

### 3.1 `workers/env/env_worker.py`

```python
class EnvWorker(Worker):                         # CPU actor;每个持 1 真 env
    def __init__(self, env_cfg: dict, task_id: int,
                 replay: "ray.ActorHandle") -> None: ...   # 持 ReplayWorker 句柄
    def init(self) -> None: ...
        # 在 actor 进程内创建 env(spawn 安全);env.set_task(task_id);obs,_=env.reset(episode_id=0)
        # self.obs / self.episode=[] / self.episode_id=0 / self.step_i=0
    def step(self, action, obs_embedding):       # action: 7 维 policy_action;obs_embedding: action_hidden
        # next_obs, reward, term, trunc, info = env.step(action)
        # tr = env.make_transition(self.obs, action, reward, term, trunc, info)
        # tr["obs_embedding"] = obs_embedding;  self.episode.append(tr)
        # done = term or trunc
        # if done: replay.add_episode.remote(self.episode); self.episode=[]; 换 episode_id;
        #          self.obs,_ = env.reset(episode_id=..., task_id=self.task_id)
        # else:    self.obs = next_obs
        # return (self.obs, done, info)           # 返回下一 obs 供 S3 推理
    def current_obs(self): return self.obs        # 供首帧/重连
```
- **auto-reset**:`done` 时本地重置(`episode_id += 1`,同 task);跨 task 轮转留 §7。
- **`make_transition` 的 obs 是 pre-step obs**,故 `self.obs` 必须在 step 间保持。

### 3.2 `workers/replay/replay_worker.py`

```python
class ReplayWorker(Worker):                      # CPU/主存 actor;共享单例
    def __init__(self, replay_cfg: dict) -> None: ...
    def init(self) -> None: ...
        # self.replay = OnlineReplay(**replay_cfg)   # 参数透传(capacity/sequence_length/task_ids/
        #   capacity_mode/failure_prefix_*/task_balanced/rank)
    def add_episode(self, episode: list[dict]):  return self.replay.add_episode(episode)   # 返回 record|None
    def sample(self, batch_size: int):           return self.replay.sample(batch_size)     # dict[str, Tensor]
    def size(self):                              ...   # 当前 episode/可采样量
    def ready(self, min_episodes: int) -> bool:  ...   # 供 learner 判断是否够采(S4)
```
- `sample` 返回 torch tensor,经 Ray object store 序列化回 learner(S4)。单机=plasma 共享内存,廉价。

### 3.3 数据流(本期,dummy 驱动)

```
for _ in range(steps):
    for w in env_workers:                         # k 个 actor 并发
        obs = w.current_obs()                     # 或上轮 step 返回
        action, hidden = dummy_policy(obs)        # 本期 dummy(S3 换成 InferenceWorker)
        obs, done, info = w.step(action, hidden).wait()
# 断言:ReplayWorker.size() 增长;sample(bsz) 字段/形状正确
```

---

## 4. 数据契约(transition 字段,逐字段对齐现有)

每个 step dict(`env.make_transition(...)` + 注入 `obs_embedding`)含:

| 字段 | 来源 |
|---|---|
| `image`(uint8)、`state`(f32) | obs |
| `action`/`wm_action`/`policy_action`(f32) | 动作(wm 尺度 / policy 归一化尺度) |
| `reward`/`done`/`discount`(f32) | env.step |
| `is_first`/`is_terminal`/`is_last` | 首步 / terminated(成功)/ done |
| `task_id`(int)/`step`(int)/`task_description`(str) | env |
| `obs_embedding`(f32) | **S3 推理产出**,本期 dummy 占位 |

`OnlineReplay.add_episode` 据此算 record:`{episode, episode_id, collection_index, task_episode_index,
rank, task_id, success, length, finish_step}`;`success` 由"任一步 success / `is_terminal>0.5` /
`reward>0`"判定。→ **EnvWorker 只要字段齐、首末标志对,身份元数据自动正确。**

---

## 5. 测试 / 验收(真 ray;env 用轻量契约假 env,LIBERO 真负载留 S5)

> 原则同 S1:真实测试推 S5;S2 只证"采集侧数据流 + 格式"对。用一个**满足 env 契约的轻量假 env**
> (`set_task`/`reset`/`step`/`make_transition`/`full_record`,产合规但合成的 obs/transition),
> 避免拖 LIBERO/mujoco;**ray actor 与 ReplayWorker 都是真的**。

`tests/e2e_tests/test_s2_env_replay.py`:
1. 1 个 `ReplayWorker`(小 capacity / sequence_length)+ k 个 `EnvWorker`(假 env,固定 episode 长度)。
2. 跑到若干 episode `done` → `ReplayWorker.size()` 增长;`sample(bsz)` 返回 dict 含
   `images/obs_embedding/actions/.../task_ids/episode_success/...`,形状 `[bsz, seq_len]`。
3. **格式等价**:把同一批合成 episode 直接喂裸 `OnlineReplay` 与喂 `ReplayWorker`,`add_episode`
   返回的 record(去掉 rank/index 单调量)逐字段相等。
4. `task_id`/`episode_success` 在 record 与 `sample` 输出里正确。

**验收**:1–4 全过;`ruff`/类型检查过。

---

## 6. 实现顺序(TDD)

1. 轻量假 env(测试夹具,满足契约)。
2. `ReplayWorker`(`OnlineReplay` 透传)+ 格式等价测试。
3. `EnvWorker`(step/auto-reset/episode 攒批/push)。
4. `test_s2_env_replay.py` 串起 k worker + replay。

---

## 7. TODO(更激进版本 / 后续,本期不做)

- **分片 / 多节点 replay 服务**:per-node 或 sharded `ReplayWorker`,多机 env worker 共享写入;
  多节点阶段做(接口:`ReplayWorker` 句柄已可被多 actor 共享,升级为分片不改调用方)。
- **共享内存 IPC**:obs/episode 走 RLinf `ShmemVectorEnv` 风格共享内存 buffer(`venv.py` 的
  `_setup_buf`/`ShArray`),仅当吞吐 profile 显示 IPC 占比高;难点 = 变长 `states`/`init_state` padding。
- **逐步 streaming push**:`done` 整条 push → 改为每步/每 chunk 流式入 replay,配合更细粒度重叠(S5 重叠深化时评估)。
- **跨 task 轮转 / work-list**:EnvWorker 按任务工作表轮转多 task(仿采集器),而非单 task auto-reset。
- **per-step 身份索引**:若将来需按 step 检索 `task_id`/success,给 `OnlineReplay` 加 per-step 索引。

---

## 8. 风险

- **假 env 失真**:轻量假 env 与真 LIBERO 行为差异 → S2 测格式、S5 用真 env 兜底端到端。
- **`obs_embedding` 占位**:本期 dummy,S3 接上后字段语义才完整;契约已固定(f32 数组),S3 不应改形状。
- **ReplayWorker 串行瓶颈**:单 actor 读写串行;单机够用,吞吐问题留 §7 分片。
- **spawn 内建 env**:LIBERO/mujoco 句柄须在 actor 进程内创建(spawn 安全),不能跨进程传句柄。
