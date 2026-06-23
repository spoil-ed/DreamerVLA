# 设计(S3):InferenceWorker —— 在线 rollout 推理 Ray 化(仿 RLinf workers/inference)

- 日期:2026-06-17
- 状态:子项目 spec,待 review(第 3/5 份)
- 主题:落地总览 §7 的 **S3**:把在线 rollout 的**动作选择前向**做成一个 GPU actor
  `InferenceWorker`,复用现有 `_rollout_action` 路径(`encoder` 冻结 + `world_model` + `policy`);
  与 EnvWorker(S2)经 channel 组成 env↔infer rollout 闭环。
- **修正(据勘探)**:在线 rollout 推理是 `_rollout_action`(`online_cotrain_runner.py:184-217`),
  **不是**离线采集器的 `OFTRolloutHiddenExtractor`。本 spec 的复用与 parity 对象=`_rollout_action`。
  (总览 §4/§10 的 `OFTRolloutHiddenExtractor` 引用据此更正。)
- 范围:**仅 S3**。InferenceWorker + env↔infer 闭环。learner/权重回灌是 S4,重叠调度是 S5。
- 关联:
  - S1 骨架、S2 `EnvWorker`(产 obs、消费 action + obs_embedding)
  - `dreamervla/runners/online_cotrain_runner.py:184-217`(`_rollout_action`,parity 对象)
  - `dreamervla/runners/online_utils.py:115-181`(`obs_to_action_hidden` / `obs_to_input_token_embedding`)
  - 模块:`self.encoder`(冻结)、`self.world_model`、`self.policy`
- 参考实现(RLinf):`RLinf/rlinf/workers/inference/`(只读推理副本)

---

## 1. 范围与边界

- **范围内**:`workers/inference/inference_worker.py`(`InferenceWorker`);K 路批量前向 + 每 env
  独立 recurrent 状态;env↔infer 闭环(channel gather/scatter)。
- **范围外**:learner 反传与权重 push(S4;但 S3 预留 `update_weights` setter)、重叠调度(S5)、
  input_token_embedding 模式(§7 TODO,默认 action_hidden)。

---

## 2. 目标 / 非目标

**目标**
1. `InferenceWorker` 复现 `_rollout_action` 的前向:`encoder`(冻结)→`obs_embedding`(action_hidden)→
   `world_model`(`encode_latent`/`observe_next`/`actor_input`)→feat→`policy`(`sample`,`return_chunk`)→
   action chunk;输出 `(policy_action, obs_embedding[, latent])`。
2. **K 路批量前向**,每 env 持**独立 recurrent 状态**(`latent`/`prev_action`/`is_first`),互不串扰。
3. 与 S2 EnvWorker 经 channel 闭环:gather k obs → 批量前向 → scatter k action+hidden。
4. **parity**:同 obs+seed 下与单机 `_rollout_action` 数值一致(§5)。

**非目标**
- 不拆 encoder/world_model/policy 为多 actor(决策 D4:串行链,拆只增序列化)。
- 不训练任何模块(纯推理;权重由 S4 同步进来)。

---

## 3. 模块与 API

### `workers/inference/inference_worker.py`

```python
class InferenceWorker(Worker):                   # GPU0 actor
    def __init__(self, model_cfg: dict, init_ckpt: dict, num_envs: int) -> None: ...
    def init(self) -> None: ...
        # 加载 encoder(eval/冻结)、world_model、policy(从 warmup/init ckpt);
        # self.state = [ {latent:None, prev_action:None, is_first:True} for _ in range(num_envs) ]
    def reset_states(self, env_ids: list[int]) -> None: ...   # episode 重置时清对应 env 的 recurrent 状态
    def forward_batch(self, obs_batch: list, env_ids: list[int]):
        # 对每个 (obs, eid):encoder->obs_embedding;
        #   world_model encode_latent(首步) / observe_next(用 state[eid].latent/prev_action);
        #   actor_input->feat;policy sample(return_chunk)->action_chunk;
        #   更新 state[eid].latent / prev_action / is_first=False
        # 批量执行(K 个 obs 一次前向),返回 list[(policy_action, obs_embedding)]
        return actions, hiddens
    def update_weights(self, world_model_sd, policy_sd) -> None: ...   # S4 权重回灌入口(本期可空载测试)
```
- **batched + per-env state**:批维 K 一次前向,但 `world_model.observe_next` 的 recurrent 输入按
  `env_ids` 取各自 state;借鉴迁移 spec §5.1"K 个独立 history buffer"。
- `obs_embedding` 默认走 `obs_to_action_hidden`(target_token 隐藏态);形状与 S2 契约一致。

### env↔infer 闭环(配 S5 runner;S3 自测用最小驱动)

```
obs_batch = [w.current_obs() for w in env_workers]          # gather
actions, hiddens = infer.forward_batch(obs_batch, ids).wait()
for w, a, h in zip(env_workers, actions, hiddens):
    obs, done, info = w.step(a, h).wait()                   # scatter;done -> infer.reset_states([id])
```

---

## 4. 数据流要点

- gather/scatter 经 `Channel`(S1):EnvWorker put obs → InferenceWorker `get_batch(k)` → 批量前向 →
  put k 个 action → EnvWorker get。S3 自测可先用直接句柄调用,channel 化在 S5 接。
- episode `done` → 通知 InferenceWorker `reset_states([env_id])`(清 recurrent),与 EnvWorker 的
  auto-reset 对齐。

---

## 5. 测试 / 验收(真 ray + 真模型;parity 是 S3 的核心)

> parity 判据(决策):**`allclose` 数值容差**(批量前向改归约顺序,逐位通常不可得);**B=1** 时
> 尽量逐位作附加 check;decoded action 必须一致。

`tests/e2e_tests/test_s3_inference_parity.py`:
1. **B=1 parity**:固定 seed,一段 obs 序列分别过单机 `_rollout_action` 与 `InferenceWorker.forward_batch`
   (B=1)→ `policy_action`/`obs_embedding` `allclose`(容差内);decoded action 相等;B=1 尽量逐位。
2. **K 路无串扰(partner-invariant)**:把 env i 的同一序列分别在"单独跑"与"混在 K 批里跑"两种情形下
   前向,env i 的输出 `allclose`(drift≈0)→ 证明批内各 env 状态隔离、无跨样本泄漏。
3. **recurrent reset**:`reset_states([i])` 后 env i 回到首步语义(`is_first` 路径),不带旧 latent。
4. `update_weights` 空载:传入当前 sd,前向不变(为 S4 留接口)。

**验收**:1–4 全过(需 GPU + 真 ckpt;heavy,标记 slow)。

---

## 6. 实现顺序(TDD)

1. `InferenceWorker.init`(载模型 + 初始化 K 状态)。
2. `forward_batch`(B=1 先对齐 `_rollout_action`)→ parity 测试 1。
3. K 路批量 + per-env state + `reset_states` → 测试 2、3。
4. `update_weights` setter(空载)→ 测试 4。

---

## 7. TODO(更激进版本 / 后续,本期不做)

- **严格逐位 parity 模式**:强制 B=1 串行以求逐位相等(牺牲批量吞吐);作可选校验模式。
- **推理吞吐优化**:batched KV-cache / 编译、`async_op` 前向、**多 InferenceWorker 副本**横向扩推理
  (placement 已支持多 GPU)。
- **input_token_embedding 模式**:支持 `obs_to_input_token_embedding`(非默认),需与下游 WM/数据契约对齐。
- **开环 chunk 推理**:每 chunk 一次前向(而非逐帧),配合开环采集 8× 省推理(属数据契约改动,跨子项目)。

---

## 8. 风险

- **批量 vs 逐帧数值差**:批维/归约顺序致非逐位 → 用 `allclose` + decoded action 一致守住语义。
- **per-env state 串扰**:K 路 `observe_next` 若误用共享 buffer 会跨样本泄漏 → 测试 2 专守。
- **显存**:encoder+world_model+policy 同卡 + K 批;K 过大可能 OOM,smoke 用小 K。
- **权重版本**:`update_weights` 与正在进行的 `forward_batch` 并发(S5 重叠)时需原子换权,S5 守。
