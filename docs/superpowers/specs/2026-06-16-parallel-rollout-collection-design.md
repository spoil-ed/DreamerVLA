# 并行 Rollout 采集器（解耦式，喂离线 World Model + Classifier）

- 日期：2026-06-16
- 状态：设计已确认并按 `data/datasets/libero` 实测数据对齐修订，待写实现计划
- 修订：落盘 schema 改为对齐 LIBERO 实测量（维度随场景、dtype/嵌套一致）；
  分辨率/reward/元数据决策落定（见 §2、§5）；classifier 收敛为单一数据集类（见 §5）。
- 相关代码：`dreamervla/runners/collect_online_rollouts_for_classifier.py`、
  `dreamervla/envs/train_env.py`、`dreamervla/dataset/libero_balanced_terminal_dataset.py`、
  `dreamervla/dataset/libero_pixel_rynn_hidden_sequence_dataset.py`、
  `dreamervla/runners/eval_libero_vla_runner.py`、`dreamervla/dataset/online_rollout_dumper.py`

## 1. 背景与目标

某些任务只有 one-trajectory 源数据，不足以训练 world model（WM）和 success
classifier。目标：用已有的 **one-trajectory VLA 作为初始 policy**，在环境
（LIBERO，后续可扩展 CALVIN）里**并行**采集大量 VLA rollout 轨迹，dump 到磁盘，
再用**现有离线训练入口**训练 WM 和 classifier。

两条分支：

- **冷启动**（任务数据不足）：one-traj VLA → 并行采集 → dump → 现有
  `train_wm.sh` / 离线 classifier 训练。
- **热启动**（已有数据已足够）：直接走现有离线管线训练 WM / classifier，
  **不需要新代码**（复用 `train_wm.sh` 指向已有 `hdf5_reward_dir` + `action_hidden_dir`）。

因此本设计的真正交付物 = **并行采集器 + WM 可训练的 dump 格式（对齐 LIBERO 实测 schema）
+ 配置/脚本 + 一个收敛的 classifier 数据集类（继承现有基类，读同一份落盘）**。

## 2. 关键事实（已核实）

- **env 不自己跑 VLA**：`train_env.py:9` 明确观测里携带 PIL 历史与 VLA record，
  调用方在外部跑 encoder+policy，把动作传给 `env.step()`。采集器完全控制动作来源。
- **现有采集器需要已训练 WM**（`collect_online_rollouts_for_classifier.py`：
  `feat = world_model(actor_input)` → `policy(hidden=feat)`，`--world-model-ckpt` 必填）。
  冷启动还没有 WM，存在鸡生蛋；因此冷启动采集必须直接跑 **base 一轨 VLA 的 action
  head**（obs→encoder→action head→action），不经过 WM。
- **eval rollout 是严格单进程**（`eval_libero_vla_runner.py:14,66`），没有现成的多
  进程/多线程并行可复用。可复用的是 eval 的 **VLA 动作推理路径**（obs→action），而非
  其并行方案。
- **离线 WM 训练数据集** = `LIBEROBalancedTerminalDataset`
  （`oft_world_model_dinowm_chunk.yaml` → `worldmodel/openvla_oft_action_chunk.yaml`）。
  继承链：`BaseDataset` → `LIBEROPixelSequenceDataset`（读图/动作/奖励）→
  `LIBEROPixelRynnHiddenSequenceDataset`（读 hidden 侧车 + 校验）→
  `LIBEROBalancedTerminalDataset`（正负窗平衡 + reward_mode）。
  - **它实际从源 HDF5 只读这几样**（`libero_pixel_sequence_dataset.py:164-205`，实测）：
    `data/<demo>/obs/<image_keys>`（默认 `agentview_rgb`、`eye_in_hand_rgb`，uint8）、
    `data/<demo>/actions`、`data/<demo>/rewards`、`data/<demo>/dones`；balanced 层再读
    `data/<demo>/sparse_rewards`（无则回退 `rewards`）。**不读** `states`/`robot_states`/
    proprio 子键——那些只在"原始 LIBERO drop-in 保真"意义上需要（见 §5）。
  - **图像分辨率与训练解耦**：dataset 自带 `_resize_images(image_size)`，把读到的任意
    分辨率统一缩到 `task.image_size`（实测 libero_goal=**64**）。故源 HDF5 给 128 或 256
    都能训通；这决定了 §5 的落盘分辨率是"取最自然值"而非"必须 256"。
  - **实测 `hdf5_reward_dir` 落盘**（`..._remaining_reward/*.hdf5`）：图像 **256×256** uint8；
    `actions`(T,7)f64、`dones`(T,)u8、`rewards`(T,)**f32**（dense/progress）、
    `sparse_rewards`(T,)u8、`robot_states`(T,9)f64、`states`(T,**场景维**)f64、obs 子组 7 键；
    demo.attrs 仅 `reward_scheme`/`reward_success`/`reward_success_index`，data.attrs 为空
    （预处理已丢弃原始 `model_file`/`init_state`/bddl 等元数据）。
  - **同名侧车** `action_hidden_dir`：`data/<demo>/obs_embedding` (+`action_hidden_states`)
    + `preprocess_config.json`，严格校验 model_path / action_head_type / obs_hidden_source /
    prompt_style / history / include_state / rotate_images_180 / time_horizon
    （`libero_pixel_rynn_hidden_sequence_dataset.py:145-238`）。实测 OFT 侧车：
    `obs_embedding`(T,**229376**)f16 = 56×4096，`action_hidden_states`(T,56,4096)f16。
  - `sequence_length: 36`（H + N*K + 1）；`DEFAULT_HIDDEN_KEY = obs_embedding`。
- 现有采集器只 dump 精简 classifier schema（无 images/state，只有
  actions/dones/rewards + obs_embedding），**不能直接喂离线 WM 训练**。
- **对齐基准 = 原始 LIBERO schema**（用户指定 `data/datasets/libero`，h5py 实测 4 个 suite）：
  每个 `data/<demo>/` =
  - 顶层：`actions`(T,7)**f64** / `dones`(T,)**u8** / `rewards`(T,)**u8** /
    `robot_states`(T,9)f64 / `states`(T,**S**)f64；
  - `obs/` 子组：`agentview_rgb`、`eye_in_hand_rgb`（T,**128,128,3**,u8）、
    `ee_pos`(3)、`ee_ori`(3)、`ee_states`(6)、`gripper_states`(2)、`joint_states`(7)，均 f64；
  - demo.attrs：`init_state`（**ndarray，= states[0]，维度 S**，非字符串）、
    `model_file`（整段 mujoco XML，~91KB）、`num_samples`（=T 的字符串）；
  - data.attrs：`bddl_file_name`、`env_args`(JSON)、`env_name`、`macros_image_convention`、
    `num_demos`、`problem_info`(JSON)、`tag`、`total`。
- **`states`/`init_state` 维度 S 随场景变化**（实测：libero_goal **79**、object **110**、
  spatial **92**、libero_10 **45**）。⚠️ **采集器绝不能硬编码 S**，必须原样取
  `env.sim.get_state().flatten()` 的长度。（旧版 spec 写死 92 是错的。）
- **三套分辨率并存、互不冲突**（实测）：① env 渲染 / 落盘 = **256**（`resolution=256`，
  与 reward dir 一致）；② 喂 OpenVLA-OFT 模型 = **224**（`image_resolution:224`，由 OFT
  自带 `resize_image_for_policy` 做 256→224 lanczos3+antialias）；③ WM 训练再降到
  **64**（`image_size`）。原始 raw 是 128，但因 WM 自带 resize，落盘取 256 最自然
  （见 §5 决策）。"对齐"= 键/嵌套/维度(含场景维)/dtype 与原始一致。
- **env 当前不暴露全字段但可还原**（`train_env.py:478-560`、`libero_env.py:198-224`）：
  `_format_obs` 只吐双相机图 + **8 维 `state`**；但 `self._last_obs`（robosuite 原始 obs）
  与 `self.env.sim` 可访问：`states(场景维 S)=env.sim.get_state().flatten()`、
  proprio 来自 `robot0_eef_pos/eef_quat/gripper_qpos/joint_pos`、
  `init_state=self.initial_states[idx]`（reset 选中那条）均可取。**需扩展 env 记录模式**
  （§8 交付物 2）。⚠️ `ee_states`(6)/`robot_states`(9) 的精确语义须实现时按 LIBERO 原始
  采集定义逐字段核对（见 §10 风险）。

- **VLA 输入维度**（实测 OFT-legacy 路由，`configs/task/libero_goal.yaml` + `train_env.py`）：
  采集器喂 VLA 与落盘给 WM 是**两套不同维度**，每步都要产：
  - proprio/state = **8** = `eef_pos`(3) + `quat2axisangle(eef_quat)`(3) + `gripper_qpos`(2)
    （`train_env.py:519-528`，`use_proprio/include_state=true`，经 proprio_projector 进 VLA）；
  - 图像 = 2 相机 × history **2** = `num_images_in_input:4`，OFT 自带 resize 到 **224×224**
    （`center_crop=true`，lanczos3，对齐 RLDS builder；源取 env 256×256）；
  - prompt = task_description（`prompt_style: vla_policy`）；action chunk horizon/`chunk_size`=8，
    action_dim=7；
  - 输出 hidden（落侧车）：`action_hidden_states`(T,56,4096)f16、
    `obs_embedding`(T,229376)f16（56=horizon8×action_dim7，即 `[8,7,4096]` 展平）。
  - 喂 VLA 复用现成 `OpenVLAOFTObsActionPolicy`（吃 `{full_image,wrist_image,state}`，
    其 docstring 明示调用方须先用 OFT `resize_image_for_policy` 准备好 224 图）。
  > RynnVLA-legacy 路由不同：encoder `target_size=256`（`rynnvla_encoder.py:177`），喂模型 256；
  > 余同，由 one-traj VLA ckpt + task `expected_*` 决定。

## 3. 总体数据流

```
[冷启动] one-traj base VLA (无 WM)
            │  并行采集 (M 卡 × K env worker)
            ▼
   WM-trainable rollout dump (磁盘)
   ├─ 源 HDF5  data/demo_*/{actions,dones,rewards,sparse_rewards,robot_states,states,
   │             obs/{agentview_rgb,eye_in_hand_rgb,ee_pos,ee_ori,ee_states,gripper_states,joint_states}}
   │             + demo.attrs[init_state] + data.attrs[env meta]   ← 对齐原始 LIBERO schema
   └─ 侧车      data/demo_*/obs_embedding  + preprocess_config.json
            │
            ├─▶ train_wm.sh (现有 LIBEROBalancedTerminalDataset, 零改动) ─▶ WM ckpt
            └─▶ classifier 训练 (单一数据集类, 继承基类读同一份落盘) ─▶ classifier ckpt

[热启动] 已有数据够 ─▶ 直接 train_wm.sh / classifier (现有路径, 零新代码)
```

## 4. 并行采集器（混合 M 卡 × K worker，config 驱动）

**方案 A（已选）**：`torchrun` 起 M 个进程（每卡一个 rank）；每个 rank 内用
`SubprocVecEnv` 风格起 K 个 CPU env 子进程并行 `step`，VLA 推理在该卡上 **batched**
（K 个 obs 一次前向）。

- **分片**：按 rank 切分 `task_ids`（以及大 suite 内的任务），每个 rank 写自己的
  shard 文件，结束后无需合并（同名侧车按 demo_key 对齐，文件名带 rank 前缀避免冲突）。
- **VLA 动作推理**：OFT 路由复用现成 `OpenVLAOFTObsActionPolicy`
  （`diagnostics/openvla_oft_obs_action_policy.py`，`policy(obs, task_description)→actions`）。
  worker 每步：env 渲染 256 → 用 OFT 自带 `resize_image_for_policy` 备好 224 的
  `full_image/wrist_image` + 8 维 `state` → policy 出 action chunk；hidden 经
  `extract_action_hidden` 路径取（落侧车）。RynnVLA-legacy 路由走 encoder 的
  `obs_to_action_hidden` + action head（同现采集器）。batched：K 个 obs 一次前向。
- **无中央 learner**：不采用 `online_dreamervla_multiproc.py` 的消息传递方案
  （有中央瓶颈、未接入）。按 rank 分片 + 各写各的，最简单、无锁、可断点续采。

**硬件与默认值**（确认：2 卡 × 80G，用 80%）：

- `collect.num_gpus (M) = 2`
- `collect.envs_per_gpu (K) = 8`（默认；2×8=16 并行 env，可按显存上调）
- 用 `torch.cuda.set_per_process_memory_fraction(0.8)` 卡住每卡 80% 显存。

**配置项**：`task_suite_name`、`task_ids`（或 `all`）、`episodes_per_task`、
`episode_horizon`、`deterministic`、`num_gpus`、`envs_per_gpu`、`seed`、输出目录。

> 备选 B（未选）：纯 DDP 每卡 1 env，改动最小，但 LIBERO 是 CPU 瓶颈，单卡吞吐上不去。

## 5. Dump 格式（已定：对齐 reward dir 的 LIBERO drop-in，单一 schema）

**决策（用户确认）**：

- **分辨率/reward 基准 = reward dir 兼容（方案 1）**：图像 **256×256** uint8；同时产
  `sparse_rewards`(T,)uint8（终止成功帧=1，其余 0）与 `rewards`(T,)（与
  `reward_mode∈{sparse,per_window_dense,from_hdf5}` 全兼容）。→ `train_wm.sh` 零改动消费。
- **维度随场景**：`states`/`init_state` 用 `env.sim.get_state()` 原生维度 S，**不硬编码**；
  proprio 各子键维度/ dtype 与原始一致（见下）。
- **元数据 = init_state + env meta，跳过 model_file**：带可复现/再处理所需的轻量元数据，
  不带 ~91KB/demo 的 mujoco XML。
- **喂 VLA 的 224 图像走 OFT 自带 `resize_image_for_policy`**（lanczos3+antialias，对齐
  RLDS builder），与落盘的 256 图像分两路，互不污染。

采集器每条轨迹写出，作为 reward dir / 原始 LIBERO 的 drop-in：

- **源 HDF5**（`data/<demo>/` 布局，键/嵌套/维度/dtype 与实测一致）：
  - 顶层：`actions`(T,7)f64、`dones`(T,)u8、`rewards`(T,)、`sparse_rewards`(T,)u8、
    `robot_states`(T,9)f64、`states`(T,**S**)f64；
  - `obs/` 子组：`agentview_rgb`、`eye_in_hand_rgb`（T,256,256,3,u8）、
    `ee_pos`(3)、`ee_ori`(3)、`ee_states`(6)、`gripper_states`(2)、`joint_states`(7)，f64；
  - demo.attrs：`init_state`（ndarray，维度 S，= reset 选中的 `initial_states[idx]`）、
    `num_samples`(=T)；
  - data.attrs：`bddl_file_name`、`env_args`、`env_name`、`problem_info`、
    `macros_image_convention`、`tag`、`num_demos`、`total`（从 task/bddl/suite 取）。
- **字段来源**（rollout 时从 env 取，见 §2 与 §10）：图像/proprio 来自 robosuite 原始 obs，
  `states=env.sim.get_state().flatten()`，`init_state` 来自 reset 选中的 initial_state。
  **需给 env 加"全量记录"模式**（§8 交付物 2）。
- **同名 action-hidden 侧车**：`data/<demo>/obs_embedding`（+`action_hidden_states`）+
  **自动生成 `preprocess_config.json`**，字段取自 task 的 `expected_*`
  （action_head_type、obs_hidden_source=action_query、prompt_style=vla_policy、history=2、
  include_state=true、rotate_images_180=true、time_horizon、model_path 等），通过
  `_validate_hidden_sidecar` 严格校验。

效果：**`train_wm.sh` 直接指向采集目录即可，零改动**；产物同时是原始 LIBERO 格式的 drop-in，
可被同一套预处理/评测脚本复用。

**classifier：收敛为单一数据集类（用户指令）**——不写适配器、不另写精简分片、不分裂多
dataset。落盘只此一份 schema；读取端：

- **WM 训练**：直接用现有 `LIBEROBalancedTerminalDataset`，零新代码。
- **classifier 训练**：定义**一个**按任务命名的数据集类（如
  `LIBEROCollectedClassifierDataset`），**继承现有 `LIBEROPixelRynnHiddenSequenceDataset`
  基类**复用读图/读 hidden 侧车/校验逻辑，从同一份落盘派生
  `obs_embedding` 窗口 + success 标签（success 由是否到达终止成功帧 `sparse_rewards` 得出）。
  若与现有 classifier dataset 有大幅公共逻辑，则抽公共基类，二者各自薄继承。

> 备选（未选）：dump 精简 classifier 格式 / 写运行时适配器——均放弃，违背"单一 schema +
> 单一数据集类，复用走基类"的收敛原则，且无法复用现有 `train_wm.sh`。

## 6. 覆盖范围与预算（config 驱动）

- **覆盖**：纯 config——`task_suite_name` + `task_ids`。小 suite（如 libero_goal）给自身
  任务；大 suite（libero_10 / 90 / 130）给全部任务。env 已支持 `init_state_sampling:
  sequential` 顺序遍历所有 init_state（`train_env.py:440-447`），天然"不从同一初始值"。
- **预算**：`episodes_per_task: N`，总量 = N × 任务数（单 suite 10 任务、N=300 → 3000）。

## 7. 范围与非目标

- **范围内**：action_hidden（legacy）路由的并行采集 + dump + 离线训练对接。路由 config 化，
  RynnVLA-legacy 与 OFT-legacy 的 action_hidden 都可用（取决于 one-traj VLA ckpt 与
  task 的 `expected_*`）。
- **非目标**：`backbone_latent` / `input_tokens` 的在线 rollout（env 未接入该 latent，
  属已知 gap）；CALVIN（结构预留，本期只做 LIBERO）；online cotrain runner 的改动
  （本期解耦，不动 `OnlineCotrainRunner`）。

## 8. 交付物清单

1. `dreamervla/runners/collect_parallel_rollouts.py`——新并行采集器（torchrun M 卡 ×
   SubprocVecEnv K worker，batched VLA 推理，base VLA 动作，无 WM）。
2. **env 全量记录模式**：扩展 `DreamerVLAOnlineTrainEnv` / `LIBERODreamerEnv`，把
   robosuite 原始 obs + `env.sim.get_state()`（场景维 S）+ `init_state` + 各 proprio 子键
   透出，供采集器写完整 schema；`ee_states`(6)/`robot_states`(9) 须按 LIBERO 原始定义核对。
3. dump writer：产出 reward-dir 兼容的 LIBERO drop-in（源 HDF5：顶层 +`obs/`子组 +
   demo/data attrs[含 init_state、跳过 model_file] + `sparse_rewards`，256×256）+
   `obs_embedding` 侧车 + 自动生成 `preprocess_config.json`。维度随场景，不硬编码。
4. **单一** classifier 数据集类（按任务命名，继承 `LIBEROPixelRynnHiddenSequenceDataset`
   基类），从同一份落盘派生 obs_embedding 窗口 + success 标签；不写适配器/精简分片。
5. `configs/experiment/collect_rollouts_action_hidden.yaml` +
   `scripts/run_collect_rollouts.sh`（torchrun，M=2，set_per_process_memory_fraction 0.8）。
6. 文档：在 `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_action_hidden_world_model.md`
   增加"冷启动并行采集"一节。

## 9. 验收标准

1. 采集器在 2 卡 × K worker 下并行采集，单卡显存 ≤ 80%，吞吐显著高于单进程 eval。
2. 产出目录能被 `train_wm.sh experiment=oft_world_model_dinowm_chunk` **零改动**消费并跑通
   至少若干训练 step。
3. 产出目录能被离线 classifier 训练消费并跑通。
4. 覆盖：sequential 模式下每个 task 的 init_state 被均匀遍历；`episodes_per_task`
   生效，总量 = N × 任务数。
5. 侧车 `preprocess_config.json` 通过 `_validate_hidden_sidecar` 严格校验。
6. 提供 1 个低成本 smoke 配置（小 N、小 horizon、1 卡）验证端到端可执行。

## 10. 风险与待确认

- **proprio 子键语义对齐**（新增、核心）：`ee_states`(6)/`robot_states`(9) 在原始 LIBERO
  里的精确组成需按其原始采集定义逐字段核对（env 当前只吐 8 维 state，须在"全量记录"模式
  里正确拼出 6/9 维），并用一条真实 demo 数值比对验证。实现计划首步先做。
- **VLA 动作推理入口**：OFT 路由已确认可复用 `OpenVLAOFTObsActionPolicy`（吃
  `{full_image,wrist_image,state}`，调用方用 OFT `resize_image_for_policy` 备 224 图）；
  RynnVLA-legacy 走 `obs_to_action_hidden`+action head。实现首步各跑一条单轨冒烟。
- **SubprocVecEnv 与 LIBERO/mujoco 的兼容**：env 子进程化需确认 LIBERO 句柄可在子进程
  内创建（通常 spawn 模式可行），并处理各 worker 的 seed / task / init_state 分配。
- **显存与 K**：K=8 为保守默认，按实际 batched VLA 前向显存上调。
