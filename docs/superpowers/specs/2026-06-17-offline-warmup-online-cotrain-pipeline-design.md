# 设计:离线 warmup → 在线 cotrain 流水线（统一回放池预灌注）

- 日期：2026-06-17
- 状态：已批准（待 spec 复核）
- 主题：在 runner 层提供一条"训练 DreamerVLA + 对 world model / classifier 做 warmup"的流水线；warmup 使用之前采集的离线轨迹数据，随后转入在线 cotrain。
- 关联：
  - `dreamervla/runners/online_cotrain_runner.py`（复用其全部在线机制）
  - `docs/superpowers/specs/2026-06-16-rlinf-vectorized-rollout-migration.md` §10（Ray TODO，本设计实现后补充）
  - `AGENTS.md:6`、`AGENTS.md:69`（单机 torchrun 为主线，不引入 Ray stack）

---

## 1. 背景与现状

仓库已有的相关件：

- **离线 `DreamerVLARunner`**（`runner_name="joint_dreamervla"`）：在固定 replay 数据集上做
  Phase-1 WM 预训练 + Phase-2 imagination actor-critic；但**不** warmup classifier，
  对 `wmpo_outcome` 路线只从 `init.classifier_state_ckpt` 加载冻结 classifier。
- **在线 `OnlineCotrainRunner`**（`dreamervla/runners/online_cotrain_runner.py`，
  `runner_name="online_cotrain"`）：env rollout → `OnlineReplay` → WM warmup + classifier
  warmup → actor-critic cotrain，单次 Hydra 调用，按 `training.warmup_steps`（line 284）
  切相位。每个训练迭代依次跑 WM 相位（训 WM）→ classifier 相位（训 classifier）→ RL 相位；
  warmup 阶段只跑前两相位（policy/critic 冻结），cotrain 阶段追加 RL 相位
  （`dino_wmpo_outcome_step`，其内部对 WM + classifier 临时冻结）——即 WM/classifier 在
  cotrain 期间**仍持续训练**。**其 warmup 数据来自在线 replay**（早期弱策略采集），而非离线数据。它已支持从 `init.world_model_state_ckpt`
  （line 251）与 `init.classifier_state_ckpt`（line 102）热启动。
- **采集器输出**（`dreamervla/dataset/rollout_dump_writer.py`）：reward HDF5 + 同名
  hidden 侧车 + `preprocess_config.json`（详见 §4）。

### 决策

- 采用 **方案 B**：离线 warmup（WM + classifier 用之前采集的轨迹数据）→ 在线 cotrain。
- 实现采用 **方案 3（统一回放池，离线轨迹预灌注）**，理由见 §3。
- classifier warmup 默认使用**同一批离线轨迹数据**的成功/失败标签（`sparse_rewards`）。

---

## 2. 目标 / 非目标

**目标**

1. 单次 Hydra 调用产出一个可用的 DreamerVLA policy：先在冷启动轨迹上把 WM + reward
   classifier warmup 好，再在线 RL 训 policy，期间 WM + classifier 继续 cotrain。
2. 最大复用 `OnlineCotrainRunner` 的在线机制；新代码集中在"离线 HDF5 → 回放池"灌注，
   及一个薄的离线 warmup 循环。
3. warmup 产物**分开**落 `wm_warmup.ckpt` / `classifier_warmup.ckpt`，各自可复用/可续跑
   （warmup 一次，在线实验多次；可单独重训其一）。
4. 提供 config 早校验 + 低成本 smoke/e2e。

**非目标**

- 不引入 Ray / 多节点 / 集群（仅在 §9 作为未来文档项补充）。
- 不改 `OnlineCotrainRunner` 的纯在线行为（只做机械抽取以便子类复用）。
- 不改离线 `DreamerVLARunner.run()`。（采集器仅做 §4.1 的 per-demo 身份元数据补齐，
  向后兼容；不动其采集逻辑/吞吐路径。）
- 不支持 `latent_type=backbone_latent` 的在线 rollout（沿用现有 guard 报错）。

---

## 3. 为什么选方案 3（统一回放池）

对比三条候选实现：

| | 方案 1 编排 3 段（ckpt 衔接） | 方案 2 单 runner 进程内（disk-backed 离线 dataset） | **方案 3 统一回放池预灌注** |
|---|---|---|---|
| 语义一致性 | warmup 用离线 dataset、在线用 replay 采样，两套窗口/标签逻辑有漂移风险 | 同左 | **warmup 与在线同一回放池、同一 step 函数，零漂移** |
| 离线数据去留 | warmup 后丢弃，在线 buffer 从空重填 → WM/classifier 易随策略漂移 | 同左 | **离线轨迹留在 buffer 与在线混采 → 防遗忘、增覆盖** |
| 离线→在线过渡 | 三段胶水 + ckpt 往返 | 相位切换 | **就是"开始往池子里加 env episode"，一条连续过程** |
| 新代码量 | 编排 + config 组合 | 抽方法 + 离线 dataset 接线 + 两套 classifier 通路 | **仅 HDF5→transition 灌注 + 薄 warmup 循环** |
| 代价 | GPU 显存段间抖动 | 改动现有 run() 较多 | **整批离线数据进内存回放池有内存压力** |

选方案 3：在概念与训练效果上最干净——`warmup` 与 `cotrain` 的区别只是现有
`warmup_steps` 门控（RL 是否训），不是另一条数据通路。WM 学的是环境动力学（与策略
无关），冷启动轨迹永远是有效动力学样本；classifier 的 success/failure 标签亦永远有效，
故把离线数据留在 buffer 里混采是 feature 而非 bug。唯一代价（内存）按"配大回放池容量"
解决，本期接受。

---

## 4. 数据契约（采集器输出 → 回放池）

采集器（`RolloutDumpWriter`）落盘，每 `data/demo_<i>/`：

```
actions           (T, 7)         float64
rewards           (T,)           float32   — 采集器写 0
sparse_rewards    (T,)           uint8     — 终止成功步为 1，否则 0
dones             (T,)           uint8     — 末步为 1
states            (T, S)         float64
obs/agentview_rgb     (T,256,256,3) uint8
obs/eye_in_hand_rgb   (T,256,256,3) uint8
demo.attrs["init_state"]  (S,)   float64
```

hidden 侧车（独立目录，同名文件）：`data/demo_<i>/obs_embedding (T, D) float16`。
另有 `preprocess_config.json`（含 `token_dim`/`chunk_size`/`obs_hidden_source`/
`task_suite_name` 等）。

### HDF5 → transition dict 映射

`OnlineReplay.add_episode(episode)`（`online_replay.py:68`）期望每个 transition dict 含：
`image`、`obs_embedding`、`reward`、`done`、`is_terminal`、`is_last`、`wm_action`、
`task_id`、`success`。映射规则（**仅有的新逻辑**）：

| transition key | 来源 |
|---|---|
| `image` | `obs/agentview_rgb[t]`（uint8） |
| `obs_embedding` | 侧车 `obs_embedding[t]`（→ float32） |
| `reward` | `sparse_rewards[t]`（真实奖励信号；`rewards` 为 0 不用） |
| `done` | `dones[t]` |
| `is_last` | `dones[t]`（末步） |
| `is_terminal` | `bool(success_demo)` 且为末步（成功才是 terminal，超时截断不是） |
| `wm_action` | `actions[t]`（7 维） |
| `task_id` | `demo.attrs["task_id"]`（采集器新增持久化，见 §4.1） |
| `success` | `demo.attrs["episode_success"]`（采集器新增，§4.1）或回退 `any(sparse_rewards == 1)` |

灌注后回放池容量按冷启动集大小配置（`online_rollout.buffer_size`），保证离线数据在
warmup 与在线初期不被淘汰。

### 4.1 采集器元数据对齐（Phase 0 前置）

canonical LIBERO 数据把**任务身份编码在文件名**（`regenerate_libero_dataset_filter_no_op.py:128,137`，
`{task.name}_demo.hdf5`，一任务一文件）；采集器按 rank 分片、**一个 shard 跨多任务**
（`collect_parallel_rollouts.py` work-list `[(tid,ep) ...]`，写 `r0_/r1_` shard），任务级
`data_attrs` 仅在首条 demo 写一次。故采集器**无法用"一文件一任务"携带身份**，必须把身份
下沉到 per-demo。这是 `RolloutDumpWriter.write_demo` 的既有遗漏，本设计一并修复（对现有
dataset reader 向后兼容——多余 attr 被忽略，符合 migration spec §8"产出零改动消费"）。

`RolloutDumpWriter.write_demo` 新增 per-demo attrs：

- **必须**：`task_id`（int）、`episode_id`（int，canonical 的 per-task demo 序号）、
  `task_description`（str，任务语言指令——canonical 由 `task.language` 运行时携带，
  采集器 `env.task_description` 可得，`collect_parallel_rollouts.py:619`）。
- **建议（审计/复现，低成本）**：per-demo `episode_success`（bool，
  `collect_parallel_rollouts.py:365`）、`episode_horizon`（int）、file 级 `rank`（int，哪块
  GPU 采的）。

实现位点（已核实）：`write_demo` 签名（`rollout_dump_writer.py:73`）加可选参数，attrs 写在
`init_state`/`num_samples`（:167-168）之后；调用点转发已有值——单 env 路径
`collect_parallel_rollouts.py`（work-list `for task_id, ep`:606，`write_demo`:616-630）、
向量化路径 `vectorized_collect.py`（:175-183，slot 持有 task/ep）。**向后兼容已核实**：
dataset reader 只读 datasets 不读 attrs（`balanced_terminal_dataset.py:100`、基类
`PixelSequenceDataset`:166-190），新增 demo.attrs 不影响现有消费。

灌注 loader 读 `demo.attrs["task_id"]`（缺失时按 §11 回退）。

---

## 5. 架构

新增 runner：`OnlineCotrainPipelineRunner(OnlineCotrainRunner)`（`dreamervla/runners/`）。

```
run():
  cfg = ...
  build_components()                 # 复用父类逻辑（encoder/WM/policy/critic/classifier + optim）
  replay = OnlineReplay(...)
  need_wm  = not (resume 且 wm_warmup.ckpt 存在)
  need_cls = not (resume 且 classifier_warmup.ckpt 存在)
  if need_wm or need_cls:
      _seed_replay_from_offline(cfg, replay)         # §4，HDF5 → add_episode
  if need_wm:
      _offline_warmup_wm(replay, wm_warmup_steps);  save wm_warmup.ckpt
  else:
      load wm_warmup.ckpt → WM
  if need_cls:
      _offline_warmup_classifier(replay, classifier_warmup_steps);  save classifier_warmup.ckpt
  else:
      load classifier_warmup.ckpt → classifier (+ threshold)
  _online_cotrain_loop(cfg, env, replay, ...)        # 父类在线循环（抽取自现 run()），warmup_steps=0
```

三个新/改点：

1. **`_seed_replay_from_offline(cfg, replay)`（新）**：读 reward HDF5 + hidden 侧车，
   逐 demo 转 transition 列表，`replay.add_episode(...)`。
2. **`_offline_warmup_wm` / `_offline_warmup_classifier`（新，薄循环，复用现成 step 函数）**：
   - WM：`for _ in range(wm_warmup_steps): wm_batch = self._build_wm_pretrain_batch(replay.sample(bs)); world_model_pretrain_step(...)`
     （`_build_wm_pretrain_batch` 在 `dreamervla_runner.py:223` 起，dataloader-agnostic；
     `world_model_pretrain_step` 来自 `dreamervla.algorithms.dreamervla`）。
   - classifier：`for _ in range(classifier_warmup_steps): online_classifier_update_step(classifier=..., optimizer=..., replay=replay, ...)`
     （`online_classifier_update_step` 在 `online_dreamervla.py:394`，内部用
     `replay.sample_classifier_windows`，与在线阶段**同一函数同一 loss**：CrossEntropy）。
   - 不步进环境 → warmup 不受 `train_every` 拖。
3. **`_online_cotrain_loop(...)`（从父类 `run()` 机械抽取）**：将现 `OnlineCotrainRunner.run()`
   的"建模块"与"在线循环"拆为两个方法，父类 `run()` 仍 = 建模块 + 在线循环（纯在线行为
   不变）；子类灌注 + warmup 后调 `_online_cotrain_loop`，传 `warmup_steps=0`（已 warm，
   RL 从头训）。在线阶段 WM/classifier 继续训（现有 "always" 相位），RL 用
   `dino_wmpo_outcome_step`（现有）。

---

## 6. Checkpoint

- **`wm_warmup.ckpt` / `classifier_warmup.ckpt`**（新，分开存）：WM warmup 完成后存
  `wm_warmup.ckpt`（`{global_step, world_model}`）；classifier warmup 完成后存
  `classifier_warmup.ckpt`（`{global_step, classifier, classifier_threshold}`）。二者置于
  `${output_dir}/ckpt/`。resume 时分别检测、分别加载并跳过对应 warmup 段——可单独重训
  classifier 而不重训 WM，反之亦然。仅当两段都命中时跳过灌注。
- 在线沿用现有 `_save_cotrain_ckpt`（`online_cotrain_runner.py:495`），存
  `{global_step, world_model, policy, critic, classifier, classifier_threshold}` 到
  `${output_dir}/ckpt/latest.ckpt`。

---

## 7. 配置与校验

- 新 `configs/experiment/online_cotrain_pipeline_*.yaml`（组合入口）+ `configs/dreamervla/*`
  （runner 配置，`_target_` 指向 `OnlineCotrainPipelineRunner`，经 `runners/__init__.py`
  导出）。Hydra 经 `cfg._target_` 实例化（`train.py:37-88`）。
- 新 knobs：
  - `offline_warmup.data_dir`（冷启动 reward HDF5 目录）
  - `offline_warmup.hidden_dir`（obs_embedding 侧车目录）
  - `training.wm_warmup_steps`、`training.classifier_warmup_steps`
  - 在线 `training.warmup_steps=0`
  - `online_rollout.buffer_size` 配足以容纳冷启动集
- `dreamervla/config.py`（`validate_cfg`，13-29 行）增早校验：
  - `offline_warmup.data_dir` / `hidden_dir` 存在；
  - 侧车 `obs_embedding` 维度 == `world_model.obs_dim`；
  - `wm_warmup_steps` / `classifier_warmup_steps` ≥ 0；
  - 灌注后回放池非空；
  - `latent_type=backbone_latent` 在线未接线时清晰报错（沿用现有 guard）。

---

## 8. 测试

- **单测**：HDF5 → transition 映射（`reward`←`sparse_rewards`、`success`/`is_terminal`/
  `is_last` 推导、`obs_embedding` 维度、`wm_action` 7 维）；灌注后 `replay.num_transitions`
  与 `replay.sample` / `sample_classifier_windows` 可用。
- **smoke/e2e**：2-demo 小 fixture（reward + hidden 侧车）→ 灌注 → 各 2 步 WM /
  classifier warmup → 存 `warmup.ckpt` → 数步在线带 RL → 写 `latest.ckpt`。复用
  `OnlineCotrainRunner` 现有 `debug_*` 开关（`online_cotrain_runner.py:307-313`：
  `debug_total_env_steps` / `debug_warmup_steps` / `debug_min_replay` /
  `debug_max_train_updates` / `debug_episode_horizon`）。

---

## 9. Ray 方案补充（本设计实现后，写入 migration spec §10）

接 `2026-06-16-rlinf-vectorized-rollout-migration.md:237-239` 现有 Ray bullet 往下写，
**针对本条在线 cotrain loop**（policy + replay + learner）：

- **异构 worker 放置**：推理 actor（VLA encoder + WM 前向产 latent/action）、env worker、
  learner（WM/classifier/RL 反传）可放不同设备/进程，按各自吞吐独立扩缩。
- **infer-step-learner 流水线重叠**：rollout 推理、env step、learner 更新三者重叠，掩盖
  互相等待（本采集器内的 infer/step 重叠用双缓冲即可，无需 Ray；此处针对整体 loop）。
- **多节点**：replay 作为共享/分片服务，多机 env worker + 多机 learner（DDP/FSDP）。
- **立场重申**：单机 torchrun/DDP/FSDP 仍为主线（`AGENTS.md:6`、`AGENTS.md:69`）；Ray 仅
  作整体 loop 的**未来架构选项**记录，本期不实现。

纯文档，不引入依赖。

---

## 10. 验收标准

1. 单次 Hydra 调用：灌注离线轨迹 → 离线 warmup WM + classifier → 分别存
   `wm_warmup.ckpt` / `classifier_warmup.ckpt` → 在线 cotrain（RL 训 policy，
   WM/classifier 继续训）→ 写 `latest.ckpt`。
2. warmup 与在线阶段对 WM/classifier 使用**同一回放池与同一 step 函数**（零语义漂移）。
3. resume 分别命中 `wm_warmup.ckpt` / `classifier_warmup.ckpt` 时各自跳过对应 warmup 段；
   两段都命中时连灌注一并跳过。
4. 单测 + smoke/e2e 通过。
5. `OnlineCotrainRunner` 纯在线行为不变（仅机械抽取方法）。
6. 采集器 `write_demo` 持久化 per-demo `task_id`/`episode_id`（+ 建议档），向后兼容
   现有 reader（§4.1）。
7. migration spec §10 Ray bullet 已补充本 loop 的 Ray 方案。

---

## 11. 风险

- **内存**：`obs_embedding` 较大，整批离线数据进内存回放池有压力 → 按冷启动集大小配
  `buffer_size`；必要时只灌注子集（但会牺牲方案 3 的"离线数据稳定器"收益）。
- **task_id 推导**：由 §4.1 采集器持久化 `demo.attrs["task_id"]` 解决。对**无该 attr 的
  旧数据**回退：单任务用 config `offline_warmup.task_id` 给所有 demo 赋值；多任务旧数据需
  重采（按 work-list/分片重建映射太脆，不采纳）。缺 task_id 会使回放池任务统计
  （`get_replay_task_stats_global`）失真，故必须有可靠来源。
- **维度对齐**：侧车 `obs_embedding` 维度须与 `world_model.obs_dim` 一致（§7 校验）。
- **冷启动策略偏弱**：warmup 数据多为弱策略轨迹，classifier 正样本可能偏少 →
  `online_classifier_update_step` 已有类平衡逻辑，warmup 步数与 `early_neg_stride` 需配。
- **数据对齐审计（2026-06-17）结论**：采集器数据内容与 canonical 基本对齐（字段/dtype/shape/
  图像旋转/obs_embedding 来源·维度·历史/action 原始尺度/sparse_reward 终止步放置）。两个发现：
  (1) 已验 `wmpo_aligned_latent_dataset.py:79,85` 用 `rewards`（采集器恒 0）而非 `sparse_rewards`
  判 `complete`，会把采集 episode 全标为失败——**不在本方案路径**（本方案用 OnlineReplay +
  `online_classifier_update_step`，读 sparse_rewards/episode_success），但作为潜伏 bug 由 plan
  Task 1B 顺修。(2) obs_embedding 图像预处理 PIL（离线）vs TF（采集器）≈0.25 token-space 残差，
  刻意（TF 为真机路径），warmup 可接受——列运行时验证项。
