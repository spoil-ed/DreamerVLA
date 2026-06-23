# 冷启动采集器 Hydra 化 + 与 WM 消费对齐(设计)

> **归档说明(2026-06-23)**:本设计稿已从 `docs/superpowers/specs/` 移入设计史;所述 Hydra 化采集器(`CollectRolloutsRunner`)已实现。
> 产物与 discrete WM 的 dump-schema/parity 契约另见 `docs/ray_rlinf_alignment_todo.md` §3。

- 日期:2026-06-17
- 状态:**§4 已定 + 设计已批准(2026-06-17)**,待写实现计划;并行采集/批量推理/discrete 已实现见
  `docs/history/2026-06-16-rlinf-vectorized-rollout-migration.md`
- 目标:把采集器从 `key=value` argparse 改为 **Hydra 入口**,使采集与 WM 训练**读同一份 task
  config**(单一真相源),并实测产物被现有 discrete WM dataset 零改动消费。

---

## 1. 关键概念澄清:两个独立的 "history"

之前混为一谈,RynnVLA 已有配置证明它们**互不相干**:

| | 含义 | 谁决定 | 配置位置 |
|---|---|---|---|
| **A. 提取历史**(h1/h2) | VLA 算**每帧** `obs_embedding` 时堆几帧图作输入 | **VLA 提取**时定(侧车属性) | `task.X.expected_history`;侧车目录 `_h1/_h2`;dataset 用 `expected_history` **校验** |
| **B. WM 序列/rollout 规划**(H, N, K) | dataloader 把**逐帧** embedding 切成训练序列 + WM 几步闭环 rollout | **WM dataset + model config**(dataloader) | `sequence_length = H + N*K + 1`、`chunk_rollout_chunks=N`、`chunk_size=K` |

证据(`configs/worldmodel/rynnvla_action_chunk.yaml`):`sequence_length: 24 = 3 + 4*5 + 1`、
`chunk_rollout_chunks: 4`(`ChunkAwareDinoWMWorldModel` 的 N 步闭环 rollout);而 `expected_history`
只是 `${task.legacy_action_hidden.expected_history}` —— **B 完全独立于 A**。

**结论:**
- **WM 有 rollout**(N 步闭环),由 dataloader+model 规划,与提取历史无关。
- **dataloader 决定 his 规划**:同一份逐帧 embedding,WM config 想切成多少 H/N/K 都行。
- **h1/h2 不是二选一、也不是 WM 的事**——它是**提取侧的一个 task config 值**;采集器与 WM
  读同一个 `task.X.expected_history` 即自动对齐,**无需猜**。

---

## 2. action_hidden = VLA 自身的 latent

`obs_embedding` = VLA LM 在 action-token 位置的 hidden `[56,4096]` 展平(229376),是 **VLA 自己的
内部表示**。故 WM 训在 **VLA 的 latent 空间**里;冷启动自洽(动作与 embedding 都来自同一个
one-traj VLA)。`preprocess_config` 的 `model_path`/`action_head_type`/`history` 只是**记录"这批
latent 由哪个 VLA、什么提取设置产的"**,防止把不同 VLA 的 latent 混喂——这正是
`_validate_hidden_sidecar` 的全部职责。

---

## 3. 数据对齐现状(已实测)

我们的 discrete one-traj 产物 vs 现有 L1 processed_data
(`libero_goal_no_noops_t_256_pi06_remaining_reward` + `..._oft_official_legacy_action_hidden_..._h2`):

- reward HDF5 顶层(`actions/dones/rewards/sparse_rewards/robot_states/states`,states 含场景维 79)
  + `obs/` 子组(双相机 256² + ee_pos/ee_ori/ee_states/gripper/joint)+ sidecar
  `obs_embedding (229376,) f16`:**逐字段 shape/dtype 100% 一致**。
- `preprocess_config` **唯二差异**:`action_head_type`(ours `oft_discrete_token` vs `oft_l1_regression`)、
  `include_state`(ours `false` vs `true`)。**不是 bug**,正确反映"我们用 one-traj **discrete**"。

格式/schema 逐字节同构,dataloader 读两者方式完全一样,只在 `_validate_hidden_sidecar` 比对
`expected_*`。

### 现有 discrete WM 消费配置已存在
`experiment=oft_discrete_token_world_model_dinowm_chunk` → `worldmodel=openvla_oft_discrete_token_action_chunk`,
期望:`oft_discrete_token` / `include_state=false` / **`expected_history: 1`(`_h1`)** / `time_horizon=8`
/ `chunk_size=8`,reward 读 `task.openvla_oft.hdf5_reward_dir`、hidden 读
`${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h1`。

> ⚠️ 该配置 `expected_history=1`;采集器**之前硬编码 h2**。Hydra 化后采集器读 `task.X.expected_history`
> → 与此自动一致(h1 就 h1,且 `num_images_in_input = expected_history × views`)。

---

## 4. 数据决策(已定 2026-06-17)

1. **冷启动采集 ckpt + 路由 = OFT one-traj discrete。**
   `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1`(磁盘已存在,含
   `dataset_statistics.json`),即现有 `task=OpenVLA_Onetraj_LIBERO` 的 `openvla_oft.ckpt_path`。
   当前采集器即 `OFTRolloutHiddenExtractor`,与此对齐;RynnVLA 路由(`VLA_model_256/libero_goal`)需
   全新 extractor,**不在本期**(RynnVLA-002 仅作接口/参数参考,见
   `/mnt/data/spoil/workspace/Related_Work/RynnVLA-002/`)。
2. **采集产物落盘 = 复用 OpenVLA_Onetraj 命名空间(当前为空,不会覆盖)。**
   reward → `task.openvla_oft.hdf5_reward_dir`(= `…/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward`);
   hidden → `task.openvla_oft.action_hidden_dir`(discrete = `${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h1`)。
   **正是 discrete WM 在 `task=OpenVLA_Onetraj_*` 下读取的目录** → 零新路径、零对齐。

---

## 5. Hydra 化方案

当前采集器是 `key=value` argparse(违反 AGENTS.md "install/preprocess/train/wm/classifier/eval
入口都应 Hydra-centered")。改为:

1. **`CollectRolloutsRunner(BaseRunner)`**(`dreamervla/runners/collect_rollouts_runner.py`):
   - 只需实现 `run()`(setup/execute/teardown 有默认);torchrun 下读 `RANK/WORLD_SIZE/LOCAL_RANK`。
   - **所有提取参数从 `self.cfg` 读**:ckpt、`expected_history`(→ history & `num_images=history×views`)、
     `expected_action_head_type`/`expected_include_state`/`expected_obs_hidden_source`/`expected_prompt_style`/
     `expected_rotate_images_180`/`time_horizon`、reward/hidden 输出目录、采集 knobs
     (`task_ids`/`episodes_per_task`/`episode_horizon`/`envs_per_gpu`)。
   - **复用现有** `OFTBatchedDecoder` / `VecRolloutEnv` / `collect_vectorized` / `RolloutDumpWriter`
     /单进程路径(把 `main()` 主体抽成 `collect_rollouts(cfg_dict, rank, world_size, local_rank)`,
     **仅由 runner 调用**)。
   - **纯 Hydra,删除 argparse**:删掉 `collect_parallel_rollouts.py` 的 `_parse_args`/`main`;
     `_make_preprocess_config` 从 cfg 的 `expected_*`/`time_horizon` 派生,**无任何硬编码默认**(缺参即报错)。
     唯一入口 = `python -m dreamervla.train experiment=collect_rollouts_onetraj`。
2. **配置**(纯 Hydra,单源在 task 层):
   - **新 task `configs/task/OpenVLA_Onetraj_ColdStart_LIBERO.yaml`**(`defaults: [OpenVLA_Onetraj_LIBERO, _self_]`):
     覆盖 `openvla_oft` 组为 discrete one-traj 提取值 —— `expected_action_head_type: oft_discrete_token`、
     `expected_include_state: false`、`expected_history: 1`(⇒ `num_images = 1×2 视图 = 2`)、
     `action_hidden_dir: ${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h1`、`time_horizon: 8`。
     `ckpt_path`(one-traj discrete)与 `hdf5_reward_dir`(OpenVLA_Onetraj 命名空间)原样继承。
     **这是采集器与 WM 共读的唯一真相源。**
   - **`configs/experiment/collect_rollouts_onetraj.yaml`**(瘦):仅
     `_target_: dreamervla.runners.CollectRolloutsRunner` + `training.out_dir` + `collect:` 旋钮块
     (`policy_mode/task_ids/episodes_per_task/episode_horizon/envs_per_gpu`)。ckpt/目录/`expected_*` 全来自 task。
   - 采集:`python -m dreamervla.train experiment=collect_rollouts_onetraj task=OpenVLA_Onetraj_ColdStart_LIBERO`
     (torchrun M-rank 同现有 launcher;采集器仅读 `RANK/WORLD_SIZE/LOCAL_RANK` 做 Layer-1 分片,**不走 torch DDP**)。
   - 消费:`python -m dreamervla.train experiment=oft_discrete_token_world_model_dinowm_chunk task=OpenVLA_Onetraj_ColdStart_LIBERO`
     (现有 worldmodel 覆盖会**重申同一组 discrete 值** —— 一致而非冲突;不改动正在工作的 worldmodel 配置)。
3. 在 `dreamervla/runners/__init__.py` 注册 `CollectRolloutsRunner`(import + 包装 + `PUBLIC_RUNNERS` + `__all__`)。
4. **删除 argparse 入口**,`scripts/run_collect_rollouts.sh` 改为 torchrun M-rank 包一条 Hydra 命令
   (不再转发 key=value);同步更新/精简引用旧 argparse 的 launcher 测试。教程改为该 Hydra 命令。

**单一真相源收益**:同一 `task=OpenVLA_Onetraj_ColdStart_LIBERO` 同时决定**采集往哪写/`expected_*`**与
**WM 从哪读/`expected_*`**,彻底消除"采集产 discrete、WM 配置期望 L1"这类错位。

> **spec 自纠**:原 §5-3 写 `task=libero_goal`,但 `libero_goal` 带 **L1 6650** ckpt + h2,与 §4-1
> (one-traj discrete)冲突 → 运行目标改为 `task=OpenVLA_Onetraj_ColdStart_LIBERO`。

---

## 6. 落地步骤(§4 已定,可执行)

1. 抽 `collect_rollouts(cfg_dict, rank, world_size, local_rank)`;`_make_preprocess_config` 从 `cfg_dict`
   读 `expected_*`/`time_horizon`,`num_images = expected_history × 视图数`,extractor `history ← expected_history`。
   **无硬编码默认**:缺任一提取参数即报错(不再用 `.get(key, <旧默认>)` 兜底);删除 `_parse_args`/`main`。
2. 写 `dreamervla/runners/collect_rollouts_runner.py::CollectRolloutsRunner(BaseRunner)`:`run()` 读
   `RANK/WORLD_SIZE/LOCAL_RANK`,由 `task.openvla_oft.*` + `cfg.collect.*` 构 `cfg_dict`,断言自动探测
   policy mode == `expected_action_head_type`(早校验),再调 `collect_rollouts(...)`。注册到 `__init__.py`。
3. 写新 task `OpenVLA_Onetraj_ColdStart_LIBERO.yaml`(§5-2)+ 瘦 `collect_rollouts_onetraj.yaml`(仅
   `_target_` + `training.out_dir` + `collect:` 旋钮);`run_collect_rollouts.sh` 改为 torchrun 包 Hydra 命令,
   同步更新引用旧 argparse 的 launcher 测试。
4. 两个新配置 `--cfg job` dry-compose 通过(无 GPU)。
5. 单测:`collect_rollouts` 参数接线(缺参报错);`CollectRolloutsRunner` 从假 Hydra cfg 构出正确 `cfg_dict`。
6. 小 smoke(1 卡 `envs_per_gpu=2`,horizon ≥ `sequence_length`=36)→ 产物落到 OpenVLA_Onetraj 目录;
   sidecar `preprocess_config` = h1/discrete/无 state。
7. **端到端消费验证**:`experiment=oft_discrete_token_world_model_dinowm_chunk task=OpenVLA_Onetraj_ColdStart_LIBERO`
   的 `BalancedTerminalDataset` **零改动**加载采集目录,`ds[0]` + 若干 train step 跑通(§7 验收硬指标)。
8. classifier dataset 同样验证。

## 7. 验收

1. `experiment=collect_rollouts_onetraj` 经 torchrun 产出 discrete 逐帧产物,schema 同 §3。
2. `BalancedTerminalDataset`(discrete WM 配置)**零改动**加载该产物并训若干 step。
3. classifier dataset 消费通过。
4. 采集与 WM 的 `expected_history`/`expected_*`/目录**同源**,无需手对齐。
