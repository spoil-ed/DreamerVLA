# Cotrain RL 链路验证与观测计划

> **执行者注意（Codex）**：本计划分两部分——**Part A 训练前用合成/小规模数据验证链路逻辑**（CPU、快、不烧 GPU），**Part B 真训练时的观测与 go/no-go 判据**。先把 Part A 全部跑绿，再启动 Part B 的真跑。步骤用 `- [ ]` 勾选。

**目标**：在烧掉数小时真训练之前，用合成数据确认 cotrain 在线 RL 链路的"基本逻辑"是对的（actor 驱动 rollout 的动作 scale 一致、分类器能判别、判别→GRPO 非零优势→actor 拿到梯度、seed→训练立即介入），然后定义真训练时要盯的信号和中止判据。

**架构（已实现，未提交，在 working tree）**：
- **Issue A 修复**：进入在线阶段时用 warmup 离线数据**按 task 限量播种在线 replay**（`online_cotrain_runner._run_vectorized_cotrain` 之前的 `_online_cotrain_loop`），使 `ready_for_training` 在 warmup 一结束即满足、训练 burst 立即介入（之前空 replay + `all_ready` 门控导致冷启动空转）。
- **Issue B 修复**：真实 env 由**训练中的 KL/BC 约束 actor 驱动**（不是冻结 OFT base）。新助手 `OnlineCotrainRunner._actor_action_and_latent`（WM 编码→`policy.sample`）；`_rollout_action` 与向量化 egl loop `_vectorized_cotrain_rollout` 都改为 actor 驱动，OFT extractor 只用来取 `action_hidden`(obs_embedding)。

**已知风险（本计划要逐一证伪/确认）**：
1. **动作 scale**：cotrain env `action_input="normalized"`（`train_env._env_cfg_kwargs`/`policy_action_to_env_action`→`unnormalize_libero_action`）。actor 输出若不是 normalized，会被 env 双重缩放→动作乱→`rollout/success_rate` 灾难性归零。
2. **分类器退化**：上次 `cls/f1=0`（acc=1.0 但 f1=0）→ 想象 rollout 结果无方差 → GRPO `filter_zero_variance_groups` 全过滤 → `actor_loss=0`、`grad=0`。需确认在均衡 500 数据上分类器能判别。
3. **RL 信号**：`rl/returns_mean` 方差、`rl/actor_loss>0`、`rl/policy_grad_norm>0` 是否真的活。

**环境**：
```bash
cd DreamerVLA && conda activate dreamervla
export DVLA_DATA_ROOT="$(pwd -P)/data"
export HDF5_USE_FILE_LOCKING=FALSE        # 本机 FS 不支持 HDF5 文件锁，避免 errno=11 警告
# 已有 500-episode 均衡数据（0.57 成功/0.43 失败、每 task 50 个不同 init_state）：
#   data/collected_rollouts/libero_goal/{reward,hidden}/*.hdf5  (10 shards + preprocess_config.json)
# 只用 GPU 6,7（0-5 被别的任务占用）。
```

---

## Part A — 训练前：合成/小规模数据验证链路逻辑（CPU，快）

> 全部用 `pytest -q`，目标是几分钟内跑完，**不需要 GPU、不需要 LIBERO env、不需要真 VLA**。每个测试给出"构造什么数据 / 关键断言 / 期望结果 / 失败意味着什么"。

### Task A0：回归——跑通已实现修复的现有单测

**Files（已存在，验证不回归）**：
- `tests/unit_tests/test_offline_seed.py`（含 `test_seed_replay_caps_episodes_per_task`）
- `tests/unit_tests/test_cotrain_vec_rollout.py`（已更新为 actor 驱动：`_FakeWorldModel`/`_FakePolicy`）
- `tests/unit_tests/test_online_cotrain_pipeline.py`
- `tests/unit_tests/test_aggregate_progress.py`、`test_collect_rollouts_config.py`、`test_coldstart_warmup_cotrain_launcher.py`（前几轮的 collect 侧修复）

- [ ] **Step 1**：`pytest -q tests/unit_tests/test_offline_seed.py tests/unit_tests/test_cotrain_vec_rollout.py tests/unit_tests/test_online_cotrain_pipeline.py`
  期望：全 PASS。
- [ ] **Step 2**：`ruff check dreamervla/runners/online_cotrain_runner.py dreamervla/runners/offline_seed.py`
  期望：All checks passed。
- 失败意味着：actor-driven / seed 改动本身有破绽，先修。

---

### Task A1：GRPO 优势核心——"判别（结果有方差）→ 非零优势；无方差 → 零优势"

> 这是整个 RL 信号链路最核心的一环。`_group_advantage` 是纯函数，最易测，先证伪上次 `actor_loss=0` 的根因。

**Files：**
- 目标：`dreamervla/algorithms/ppo/grpo.py:110` `def _group_advantage(score, group_size, eps)`
- Test（新建）：`tests/unit_tests/test_grpo_advantage_variance.py`

- [ ] **Step 1：写测试**

```python
import torch
from dreamervla.algorithms.ppo.grpo import _group_advantage

def test_varying_outcomes_give_nonzero_advantage():
    # 一个 group 内 K=4 条 rollout，结果有方差（2 成功 2 失败）
    score = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = _group_advantage(score, group_size=4, eps=1e-6)
    assert adv.abs().sum() > 0          # 有学习信号
    assert torch.isfinite(adv).all()

def test_constant_outcomes_give_zero_advantage():
    # 组内结果全同（退化分类器的情形）→ 归一化优势为 0
    for s in (torch.zeros(4), torch.ones(4)):
        adv = _group_advantage(s, group_size=4, eps=1e-6)
        assert adv.abs().max() < 1e-5   # 没有任何梯度信号
```

- [ ] **Step 2**：`pytest -q tests/unit_tests/test_grpo_advantage_variance.py`
  期望：PASS。先读 `_group_advantage` 实际签名/形状（可能要求 `score` 是 `[B_eff]` 且 `B_eff % group_size == 0`），按真实签名把上面 `score` 构造成对的形状。
- 解读：
  - 两个测试都过 → 确认"分类器能判别→非零优势 / 退化→零优势"这条逻辑成立，**上次 `actor_loss=0` 的唯一可能原因就是分类器退化（结果无方差）**。
  - 若"varying→nonzero"失败 → GRPO 优势计算本身有 bug（更严重）。

---

### Task A2：LUMOS 整条 actor 更新——判别分类器 → `actor_loss>0`、`grad>0`

> 用 tiny 真 WM + tiny 真 actor + **stub 分类器**（直接控制想象 rollout 的成功/失败结果），跑 `dino_lumos_step`，验证整条链路在"分类器能判别"时真的产出非零 actor 梯度，并复现"退化分类器→零"。

**Files：**
- 目标：`dreamervla/algorithms/ppo/outcome.py:356` `dino_lumos_step(policy, chunk_world_model, classifier, classifier_threshold, actor_optimizer, obs, device, algorithm_cfg, optim_cfg, ref_policy=None)` → 返回含 `actor_loss`/`returns_mean`/`actor_grad_norm` 的 dict。
- Test（新建）：`tests/unit_tests/test_lumos_signal.py`

- [ ] **Step 1：先读这些以构造合法输入**
  - `outcome.py` 里 `obs` 用到哪些键（应为 `obs_embedding/actions/rewards/dones/is_first/is_terminal/is_last`，形状 `[B, T, ...]`）、`algorithm_cfg.lumos`（`chunk_size`/`episode_max_steps`/`ppo_rollouts_per_start`/`filter_zero_variance_groups`/`classifier_min_steps`）。
  - `chunk_world_model` 在想象时被调用的 `mode`（如 `imagine`/`actor_input`/`observe_next`），`policy({"mode":"sample",...})` 的返回 `(action_chunk, logp, extra)`。
  - `classifier`：被 `predict_success(...)` 调用（`latent_success_classifier.py:186`）；**stub 一个 `predict_success` 让你能精确控制每条想象 rollout 的成功标签**。

- [ ] **Step 2：写两个测试（同一 tiny WM/actor，仅 stub 分类器结果不同）**

```python
# 伪代码骨架——按上面读到的真实接口补全形状
def _tiny_setup(device):
    # 最小可跑的 WM + actor（可用项目里已有的 tiny/fake，或真模块开极小维度），
    # 一个 optimizer，一段合成 obs（B 个起点、每个 group_size 条 rollout）。
    ...
    return policy, wm, optimizer, obs, algo_cfg, optim_cfg

class _DiscriminativeClassifier(nn.Module):
    # 让组内一半 rollout 判成功、一半失败 -> 结果有方差
    def predict_success(self, *a, **k): ...   # 返回交替的 0/1（或基于内容可分）

class _DegenerateClassifier(nn.Module):
    def predict_success(self, *a, **k): ...   # 永远返回同一类 -> 无方差

def test_discriminative_classifier_gives_nonzero_actor_gradient():
    policy, wm, opt, obs, ac, oc = _tiny_setup(torch.device("cpu"))
    m = dino_lumos_step(policy, wm, _DiscriminativeClassifier(),
                               classifier_threshold=0.5, actor_optimizer=opt,
                               obs=obs, device=torch.device("cpu"),
                               algorithm_cfg=ac, optim_cfg=oc, ref_policy=None)
    assert abs(m["actor_loss"]) > 0
    assert m["actor_grad_norm"] > 0
    # returns 有方差（不是全 0/全 1）
    assert 0.0 < m["returns_mean"] < 1.0

def test_degenerate_classifier_gives_zero_signal():
    policy, wm, opt, obs, ac, oc = _tiny_setup(torch.device("cpu"))
    m = dino_lumos_step(policy, wm, _DegenerateClassifier(), 0.5, opt,
                               obs, torch.device("cpu"), ac, oc, None)
    assert m["actor_loss"] == 0.0 and m["actor_grad_norm"] == 0.0   # 复现上次现象
```

- [ ] **Step 3**：`pytest -q tests/unit_tests/test_lumos_signal.py`，期望 PASS。
- 解读：
  - `discriminative` 测试过 → **整条 actor 更新链路是对的**，只要给它一个能判别的分类器就会学。
  - `degenerate` 测试过 → 精确复现并定位上次 `actor_loss=0` = 分类器退化（无方差→`filter_zero_variance_groups` 全过滤）。
  - 若 `discriminative` 都拿不到非零梯度 → 想象/优势/PPO loss 链路有更深 bug，**必须先修，否则真训练注定不学**。

---

### Task A3：分类器在真实均衡数据上能判别（cls/f1 不退化）

> 用真 500-episode 均衡数据训分类器若干步，在留出集上量 f1，确认它不是上次那种 `f1=0` 的退化态。这是 Task A2 里"判别分类器"在真数据上能否得到的前提。

**Files：**
- 目标：`dreamervla/models/reward/build_classifier`、`dreamervla/runners/online_dreamervla.online_classifier_update_step`、`dreamervla/runners/offline_seed.seed_replay_from_offline`、`dreamervla/runners/online_replay.OnlineReplay`。
- Test（新建，标 `@pytest.mark.slow` 或单独脚本 `scripts/diag/check_classifier_discriminates.py`，因为要读真数据、可能要 1 块 GPU）：

- [ ] **Step 1：脚本逻辑**
  - `OnlineReplay(capacity=20000, sequence_length=<seq_len>, task_ids=(0..9))`。
  - `seed_replay_from_offline(replay, data_dir=.../libero_goal/reward, hidden_dir=.../hidden, default_task_id=None)`（全量，不限 cap）。
  - 用与 cotrain 相同的分类器配置 `build_classifier(...)` + optimizer。
  - 跑 `online_classifier_update_step(...)` ~1000 步（或直接复用 pipeline 的 `_offline_warmup_classifier`），**每 100 步打印 `loss/acc/f1`**。
  - 关键：**f1 必须在留出/在线 batch 上明显 > 0**（目标 `f1 ≥ 0.6`，acc 不是关键——退化态 acc 也高）。

- [ ] **Step 2：运行**（GPU 6,7 任一，或 CPU 若可行）
  ```bash
  CUDA_VISIBLE_DEVICES=6 HDF5_USE_FILE_LOCKING=FALSE python scripts/diag/check_classifier_discriminates.py
  ```
  期望：`f1` 随步数上升并稳定在 `≥0.6`。
- 解读：
  - f1 升到 ≥0.6 → 分类器在均衡数据上能判别，**Part B 真训练的 RL 信号有望复活**。
  - f1 始终 ~0（acc 却高）→ **分类器/特征/标签构造有更深问题**（不是数据量问题，因为这次数据均衡）。需排查：正负窗口采样（`early_neg_stride`）、标签来源（`is_terminal`/`episode_success`）、分类器输入特征（`obs_embedding` 是否含判别信息）。**此项不过，别启动真训练。**

---

### Task A4：动作 scale 一致性——actor 输出 ↔ env(normalized) ↔ wm_action ↔ 想象条件

> 最高风险项。确认 actor 驱动 rollout 不会因 scale 不一致导致动作乱、`rollout/success_rate` 归零。分静态（读+推理）和动态（真跑早期信号）两段。

**Files：**
- `dreamervla/models/actor/RynnVLAActionHiddenActor`（actor `sample` 输出 scale：normalized 还是 unnormalized？是否含 action-token 解码/`norm_stats`？）
- `dreamervla/envs/train_env.py:319 policy_action_to_env_action`（`action_input="normalized"`→`unnormalize_libero_action`）、`:663 info["wm_action"]=env_action`。
- `dreamervla/algorithms/ppo/outcome.py` 想象里 WM 的动作条件（actor 输出在喂 WM 前是否做了和 `wm_action` 一致的 scale 变换）。

- [ ] **Step 1（静态）**：读上述三处，明确回答并在脚本/注释里写下：
  1. actor `sample` 输出的动作在**什么 scale**（normalized [-1,1] / env-scale / 其它）？gripper 维度是连续还是已 binarize？
  2. env `action_input="normalized"` 期望输入 **normalized**；它对 actor 输出做 `unnormalize`。**actor 输出 scale 必须 == env 期望（normalized）**，否则双重缩放。
  3. replay 的 `wm_action == unnormalize(actor_output) == env-scale`；WM/想象条件用的是哪一个 scale？**想象里喂 WM 的动作 scale 必须 == WM 训练时见到的 `wm_action`(env-scale)**。
- [ ] **Step 2（静态断言测试，新建 `tests/unit_tests/test_actor_action_scale.py`）**：
  - 构造一个已知的 actor 输出张量（如全 0 或一个固定值），分别走 (a) `policy_action_to_env_action`（env 路径）和 (b) 想象里 WM 的动作预处理（imagination 路径），断言两条路径对同一 actor 输出得到**一致 scale 的 env-scale 动作**（即 round-trip 自洽，无双重缩放）。
  - 若代码里 actor↔env↔WM 三者 scale 约定不一致 → **这是真 bug，直接修**（让 actor 输出/各处变换统一到同一约定；优先方案：actor 输出 normalized，env 做唯一一次 unnormalize，想象里对 actor 输出做同样的 unnormalize 后再喂 WM）。
- [ ] **Step 3（动态，归入 Part B 的第一个红线检查）**：真跑进入在线后的**最早若干 episode**，`rollout/success_rate` 应 **≈ OFT base（~0.4–0.55）**，因为 actor 初始化自 OFT（≈base）。
  - 若早期 `rollout/success_rate ≈ 0` → **scale 仍不一致**（或 actor 初始动作被破坏），立刻停、回到 Step 1/2 修。
  - 若早期 ≈0.5 → scale OK，actor 驱动 rollout 合法。

---

### Task A5：seed → ready_for_training（Issue A 逻辑）

**Files：**
- `dreamervla/runners/offline_seed.seed_replay_from_offline(..., max_episodes_per_task)`、`dreamervla/runners/online_replay.OnlineReplay.ready_for_training(min_transitions, task_ids, min_episodes_per_task)`。
- Test：扩展 `tests/unit_tests/test_offline_seed.py`。

- [ ] **Step 1：写测试**

```python
def test_seeded_replay_is_training_ready(tmp_path):
    # 写一个含全部 10 个 task、各 ≥3 episode 的小 fixture（每条长度 > sequence_length）
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_multitask_fixture(rdir, hdir, tasks=range(10), eps_per_task=3, T=8)
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=tuple(range(10)), rank=0)
    seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir, max_episodes_per_task=3)
    assert replay.ready_for_training(min_transitions=432, task_ids=tuple(range(10)),
                                     min_episodes_per_task=1) is True
```

- [ ] **Step 2**：`pytest -q tests/unit_tests/test_offline_seed.py -k training_ready`，期望 PASS。
- 解读：过 → warmup 一结束在线 replay 即 `all_ready`，训练 burst 立即介入（真跑里表现为在线一开始 env/s 就掉下来、且 `rl/*` 立刻有值）。

---

### Task A6：actor 驱动向量化 rollout 机制（Issue B 机制）

**Files：** 已存在 `tests/unit_tests/test_cotrain_vec_rollout.py`（已用 `_FakeWorldModel`/`_FakePolicy`）。

- [ ] **Step 1**：`pytest -q tests/unit_tests/test_cotrain_vec_rollout.py`，期望全 PASS。
- [ ] **Step 2（可选增强）**：加一个断言——记录每步传给 `vec.step` 的动作来自 `_FakePolicy`（actor）而非 extractor 的 base 动作（例如 `_FakePolicy` 返回可识别的常数，断言 `vec` 收到的动作等于它）。
- 解读：确认 egl 实际跑的向量化 loop 是 **actor 驱动**、每 slot 维护独立 WM latent。

---

### Part A 收口（go/no-go 到 Part B）

- [ ] A0–A6 全绿。
- [ ] **A4 静态 scale 自洽**（若发现不一致，已直接修并补测）。
- [ ] **A3 分类器 f1 ≥ 0.6**（不退化）。
- 三者任一不过 → **不要启动 Part B 真训练**，先解决。
- 全过 → commit 这批改动（带 `--signoff`，subject 不含 `/` 或 `===`），再进入 Part B。

---

## Part B — 真训练：启动命令与观测/中止判据

### B0：启动（GPU 6,7，完整 warmup）

> 完整分类器/WM warmup（这是 RL 信号能否活的前提），在线步数 bounded（吞吐 ~0.1–0.3 env/s，别用默认 200000）。

```bash
cd DreamerVLA && conda activate dreamervla
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=6,7 bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=2 profile=multi_gpu render_backend=egl skip_collect=true \
  warmup.total_env_steps=3000 \
  > logs/cotrain_actorfix.log 2>&1 &
```
- `skip_collect=true` 复用已有 500-episode 均衡数据。
- **不覆盖** `warmup.wm_steps`/`warmup.classifier_steps` → 用 multi_gpu profile 的完整 2000+2000（A3 已确认这能训出判别分类器）。
- TensorBoard：`data/outputs/coldstart_warmup_cotrain/<时间戳>/cotrain/log/tensorboard/`。
- 读标量脚本：用 `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` 读 `events.out.tfevents.*`，取 tags：`rl/actor_loss`、`rl/returns_mean`、`rl/policy_grad_norm`、`rollout/success_rate`、`cls/f1`、`cls/acc`、`wm/loss`、`buffer/size`。

### B1：阶段性预期（按时间顺序观测）

| 阶段 | 预期 |
|---|---|
| warmup `[1/3]`/`[2/3]` | `wm_loss` 下降并趋稳；分类器 warmup 结束打印 `acc`（**另外查 `cls/f1`，应 >0**）。|
| 进入 `[3/3] ONLINE` | 打印 `seeded online replay with N warmup episodes (<= K/task) -> M transitions`（Issue A 生效）；`vectorized rollout: 4 envs, render_backend=egl`。|
| 在线最初 ~50–200 env-step | env/s 从高速掉到 ~0.1–0.3（**训练 burst 已介入**，对比"坏"状态的 6 env/s）；TB 开始出现 `rl/*`、`cls/*`、`rollout/*` 标量。|

### B2：判定性信号（这次相对上次"全 0"的关键对比）

| 指标 | 上次（坏） | 这次应看到（好） | 含义 |
|---|---|---|---|
| `rl/returns_mean` | 几乎全 0 | **有方差**（min<max，非全 0/全 1） | 想象 rollout 结果有成功有失败 |
| `rl/actor_loss` | 全程 0 | **非 0 且在变化** | actor 真的在被优化 |
| `rl/policy_grad_norm` | 全程 0 | **>0** | policy 真的拿到梯度 |
| `cls/f1` | 偶尔/经常 0 | **稳定 >0.5** | 分类器在线持续能判别 |
| `rollout/success_rate` | 恒定（冻结 base） | **早期 ≈0.5（actor≈base），随训练应有变化** | 现在测的是 actor 自己的成功率 |
| `buffer/size` | — | 持续增长（actor 在线采新数据） | on-policy 闭环在转 |

> 注意 `rollout/success_rate` 是**累计平均**（`n_success/n_episodes`），变化慢。要更敏感地看 PPO 效果，可（可选改进）改成滑窗成功率，或在日志里另记最近 50 个 episode 的成功率。

### B3：红线/中止判据（出现即停，别白烧）

- **A4 动态红线**：在线最早几条 episode 的 `rollout/success_rate ≈ 0`（actor≈base 时本该 ~0.5）→ **动作 scale 仍不一致**，停，回 Part A Task A4。
- **RL 死信号**：进入在线 + `all_ready` 满足后，跑了 ≥300 env-step 仍 `rl/actor_loss==0 && rl/policy_grad_norm==0` → 想象/优势链路或分类器有问题，停，回 Task A2/A3。
- **分类器退化**：`cls/f1` 长期 ~0（acc 却高）→ 停，回 Task A3。
- **egl 不稳**：日志出现 `Aborted/SIGABRT/Segmentation/read_pixels/egl spawn child died` 反复 → egl 渲染在 actor 驱动下不稳，先降 `online_rollout.num_envs`（如 2）或临时 `render_backend=osmesa` 隔离问题。
- **OOM**：80GB 卡上 7B VLA + 610M WM + actor 共存 OOM → 降 `dataloader.batch_size`/`training.classifier_batch_size`。

### B4：成功判据（拿到"真实结论"）

- `rl/returns_mean` 有方差、`rl/actor_loss>0`、`rl/policy_grad_norm>0` 持续若干百步 → **PPO 确实在学**（这是第一层结论，最关键）。
- 跑足够 episode 后，`rollout/success_rate`（或滑窗成功率）**相对早期 base 基线出现可辨的上升趋势** → **PPO 提升了 actor 的真实 rollout 成功率**（最终结论）。
  - 若信号活但成功率不升/下降：是 RL 调参问题（`kl_coef`、`clip_ratio_*`、`actor_bc_to_ref_scale`、`imagination_horizon`、`ppo_rollouts_per_start`、学习率），属于"逻辑对、需调参"，与本计划的"链路逻辑"无关。

---

## 附：当前 working tree 已实现但未提交的改动（供 Codex 心里有数）
- `dreamervla/runners/offline_seed.py`：`seed_replay_from_offline(..., max_episodes_per_task)`。
- `dreamervla/runners/online_cotrain_runner.py`：`_actor_action_and_latent` 助手；`_rollout_action` 改 actor 驱动；`_vectorized_cotrain_rollout` 改 actor 驱动 + 每 slot WM latent；进入在线前播种在线 replay；移除 `oft_open_loop_action` import。
- `tests/unit_tests/test_cotrain_vec_rollout.py`、`tests/unit_tests/test_offline_seed.py`：相应更新/新增。
- 前几轮已 push 到 main 的：collect 切片 / resume 偏移 / GPU 预检 / 聚合进度条 / ray init_state 多样性。本计划的 Part A 通过后再把上述未提交改动一起 commit。
