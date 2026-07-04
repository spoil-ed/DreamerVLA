# 主线收敛 · EGL 对齐 · 激进可回退废弃 — 设计规格 (SPEC)

- 日期：2026-07-04
- 分支：feat/rlinf-alignment-full-pipeline
- 主线事实源：`docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- 参考标准实现：`/mnt/data/spoil/workspace/RLinf`
- 状态：已批准，待实现（本文件是唯一事实源，loop agent 依据它推进）

---

## 0. 背景与关键更正

主线 = **OpenVLA-OFT one-trajectory LIBERO cotrain**，入口
`scripts/e2e_coldstart_warmup_cotrain_ray.sh` → `dreamervla.launchers.coldstart_warmup_cotrain`，
分阶段依次执行 collect → warmup(sync) → online cotrain(ray async) →（可选）post-step eval。

**关键更正（以代码实读为准）**：主线在线 cotrain 的 actor-update 路由是
**LUMOS**（`dino_lumos_step`，`dreamervla/algorithms/ppo/outcome.py`，由
`dreamervla/workers/actor/learner_worker.py:487` 直接调用），不是历史记忆里
"outcome.py 未激活" 的说法。废弃时必须保留这条 LUMOS 路径及其依赖。

已锁定的四个决策（来自需求澄清）：

1. 废弃机制：`git mv` 到 `archive/` 镜像路径 + 逐文件 manifest + 一键还原脚本。
2. R1 验收：两档都做（小 smoke + 真实全规模 5 步）。
3. EGL：默认三处全开 + 移植 RLinf per-worker 设备绑定，保留 osmesa 回退；不行再回 RLinf 对齐。
4. 废弃范围：废弃替代模型族 + 独立训练路 + 诊断；保留 cotrain 内用的 WM/分类器与 `_noray`/`_ray_base`。

---

## 1. 目标与验收标准

### R1 — 5 global_step 稳定运行 + SR 上升趋势 + base-VLA 同 harness 基线

- **先测原始 VLA**：用 `EmbodiedEvalRunner` 且 `eval.ckpt_kind=vla`，指向原始
  OpenVLA-OFT SFT 检查点（`init.vla_ckpt_path` / `eval.ckpt_path`，如
  `${data_root}/checkpoints/Openvla-oft-SFT-traj1/...`），在与 cotrain **完全相同**
  的评测配置（同 suite、同 `num_episodes_per_task`）下记录基线 SR。
- **再跑 cotrain 并对比**：cotrain 稳定跑满 5 个 global_step，SR 相对 base 基线呈
  **上升趋势**；判据宽松——"不为极低即视为合理"，只需非退化的上行信号。
- **两档验收**：
  - 档 A（smoke，低成本、CI 友好）：基于已存在的 `manual_cotrain_ray_tiny`，小 env 数、
    `manual_cotrain.global_steps=5`，用于保证管线端到端绿。
  - 档 B（真实信号）：真实 `real=32 / imagine=256 / step=512` 配置 +
    `manual_cotrain.global_steps=5`，用于确认真实 SR 趋势。
- **验证判据**：base SR 与 5 步 cotrain SR 均写入 `eval/` 命名空间并落盘到
  run root 的 JSONL / TensorBoard；报告中给出两者数值与趋势结论。

### R2 — cotrain 默认量纲：real=32 / imagine=256 / step=512

- 现值**已经**是 32/256/512，本条目标是"锁定 + 双写点同步 + 早校验 + 写进文档"，防漂移。
- 双写点（两处必须一致）：
  - `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`
    L26 `real_rollout_target_trajectories: 32`、
    L28 `wm_rollout_target_trajectories: 256`、
    L30 `max_steps_per_rollout_epoch: 512`。
  - `configs/scripts/coldstart_warmup_cotrain.yaml` `multi_gpu` profile
    L242/243/244（`ray_online_real_rollout_target_trajectories` /
    `ray_online_wm_rollout_target_trajectories` /
    `ray_online_max_steps_per_rollout_epoch`）。
- **早校验**（加到 `dreamervla/config.py`，与既有 L577/586-593/716-784 校验并列）：
  - `real_rollout_target_trajectories == 32`、`wm_rollout_target_trajectories == 256`、
    `max_steps_per_rollout_epoch == 512` 为**默认基线**；若被覆盖则告警但放行（不硬失败，
    以免阻断 smoke/tiny）。真正硬约束仍是既有的整除/chunk 一致性校验。
- **验证判据**：读校验分支 + compose 主线 experiment 时打印三值确认；smoke/tiny 覆盖时告警可见。

### R3 — collect / cotrain-real / eval 三处默认 EGL

- **现状**（需在改动前逐一实证，见 §5 任务 2）：
  - collect：`dreamervla/runners/collect_parallel_rollouts.py:497-498`
    `os.environ.setdefault("MUJOCO_GL","osmesa")`（**但该文件是否在主线 collect 真实
    渲染路径上存疑——主线 collect 走 `ColdStartRayCollectRunner`，需实证渲染入口**）。
  - cotrain-real：`manual_cotrain_ray_runner.py` `_render_backend()`（L1313-1320，默认
    osmesa）、`_real_render_backend()`（L1322-1330）；`multi_gpu` profile 里
    `ray_online_real_render_backend: osmesa`（coldstart yaml L239）强制 real env 走 osmesa。
  - eval：`coldstart_warmup_cotrain.py` `_post_step_eval_env()`（L1207-1219）默认 egl。
  - 所有 config 级 `render_backend` 默认值均为 osmesa（ray.yaml L11、ray_base.yaml L16、
    coldstart yaml L33）。EGL 目前是 opt-in（靠 `render_backend=egl` 覆盖）。
- **目标**：三处默认 `render_backend=egl`。
- **RLinf 对齐（根治历史 robosuite `read_pixels` SIGABRT）**：移植 per-worker 设备绑定——
  每个 env worker 在其 `env_fn`/子进程内按 shard 设置
  `MUJOCO_EGL_DEVICE_ID`（LIBERO/MetaWorld 用法，RLinf
  `rlinf/scheduler/hardware/accelerators/nvidia_gpu.py:113`、
  `rlinf/envs/metaworld/metaworld_env.py:97`）或 `EGL_VISIBLE_DEVICES`
  （CALVIN，`rlinf/envs/calvin/calvin_gym_env.py:80`），键为该 worker 的 `seed_offset`/shard id。
  DreamerVLA 已有 `dreamervla/utils/egl_device.py::apply_egl_device_regime()`
  与 `dreamervla/runners/render_device_config.py`，优先复用/扩展它们，不新造轮子。
- **统一 render 方案（单一 helper，核心设计）**：所有基于 LIBERO 的 EGL/osmesa 后端选择
  收敛到**唯一一个底层 helper**，collect / cotrain-real / eval 三处 LIBERO env 构造全部只调它，
  三处不再各自 `os.environ.setdefault`。osmesa 只是该 helper 的一个 `backend` 取值，**不再是
  另一条代码路径**。建议直接扩展 `dreamervla/utils/egl_device.py`：
  ```
  apply_libero_render_regime(backend: str, shard_id: int, gpu_pool: list[int]) -> None
      # backend ∈ {egl, osmesa}
      # egl: 设 MUJOCO_GL=egl / PYOPENGL_PLATFORM=egl，并按 shard_id 从 gpu_pool
      #      选 MUJOCO_EGL_DEVICE_ID；osmesa: 设 MUJOCO_GL/PYOPENGL_PLATFORM=osmesa
      # 零 GPU + egl → 抛既有 _ZERO_GPU_EGL_ERROR
  ```
  - **关键纪律（决定成败）**：GL 后端 env var 必须在**每个 env-worker 子进程、mujoco/robosuite
    初始化之前**设置，不能只在 launcher/父进程设一次。故"单一 helper"= 一个实现 + 在
    **每个 worker 的 `env_fn`/子进程入口最早处**用该 worker 的 shard id 调用它（对齐 RLinf
    `metaworld_env.py:97` 在 `env_fn` 内设 `MUJOCO_EGL_DEVICE_ID`）。LIBERO 用 robosuite
    `OffScreenRenderEnv`，靠"子进程内进程级 `MUJOCO_GL`+`MUJOCO_EGL_DEVICE_ID`"即可套进同一 helper。
  - **eval GPU 池**：`gpu_pool` 为 helper 参数、config 驱动；默认与 collect/train **共享**同一池，
    需要时由 eval 配置覆盖传入独立池——同一 helper，不建专用机制（简单优先）。
  - **收益**：R3 从"改三条分歧路径"变为"建一个 helper + 三处改成调用它"，SIGABRT 修复面收敛到一处，
    且可**无 GPU 单测**（断言给定 backend/shard 下设对了哪些 env var）。
- **回退与护栏**：保留 osmesa 作为**显式**回退（`render_backend=osmesa` 仍可用）；
  零 GPU 环境仍拒绝 EGL（保留 `_ZERO_GPU_EGL_ERROR`，coldstart launcher L38-43/398-400）。
- **失败预案**：若移植后 EGL 仍 SIGABRT，回退该路径为 osmesa 并参照 RLinf LIBERO
  `OffScreenRenderEnv` 具体用法（`rlinf/envs/libero/libero_env.py:166,172`、
  `rlinf/envs/libero/venv.py:46,159`）再对齐；此授权已获用户批准。
- **验证判据**：三处默认值改为 egl；`render_backend=egl` 端到端冒烟不崩（有 GPU 时）；
  无 GPU 时静态确认默认值与 device-binding 代码路径，标 GPU-GATED。

### R4 — 非主线代码全废弃（激进但可回退）

- 机制：`git mv` 到 `archive/<原相对路径>`；**绝不 `rm`**。同步维护：
  - `docs/superpowers/DEPRECATION-manifest.md`：逐文件表（原路径 → archive 路径 → 一句废弃理由 → 迁移 commit）。
  - `scripts/restore_from_archive.sh`：一键把 manifest 中任意/全部文件 `git mv` 回原位。
- 保留：cotrain 内部复用的 WM/分类器/actor 模型 + `_noray`/`_ray_base` 主线兄弟。
- **验证判据**：迁移后主线 6 个 experiment 仍能 Hydra compose；清理本次改动产生的悬空
  import；单测在 dreamervla env 下绿；`restore_from_archive.sh --dry-run` 能列全还原动作。

---

## 2. 主线白名单（KEEP — 其余一律进 archive）

### 2.1 configs/experiment（keep）
- `openvla_onetraj_libero_cotrain_ray.yaml`（→ `ManualCotrainRayRunner`，在线 async 主入口）
- `openvla_onetraj_libero_cotrain_noray.yaml`（→ `OnlineCotrainPipelineRunner`，同步 warmup + noray 兄弟）
- `manual_cotrain_ray_tiny.yaml`（tiny/debug，R1 档 A smoke 用）
- `collect_rollouts_ray.yaml`（→ `ColdStartRayCollectRunner`）
- `collect_rollouts_onetraj.yaml`（→ `CollectRolloutsRunner`，noray collect）
- `eval_libero_vla.yaml`（→ `EmbodiedEvalRunner`）

### 2.2 runners（keep）
`manual_cotrain_ray_runner.py`、`online_cotrain_pipeline_runner.py` 及其同步依赖
（`online_cotrain_runner.py`、`online_dreamervla.py`+`_online_dreamervla_checkpoint.py`
+`_online_dreamervla_dist.py`、`offline_seed.py`、`latent_classifier_runner.py` 中被复用的
`_sweep_metrics`、`online_replay.py`、`online_utils.py`）、`cold_start_ray_collect_runner.py`、
`collect_rollouts_runner.py`、`embodied_eval_runner.py`+mixins（`_embodied_eval_*_mixin.py`、
`_embodied_eval_helpers.py`、`eval_metrics.py`）、支撑（`base_runner.py`、
`render_device_config.py`、`real_eval_schedule.py`、`action_chunk_queue.py`、
`oft_collect_common.py`）。

### 2.3 algorithms（keep）
`ppo/outcome.py`（LUMOS）、`ppo/__init__.py`、`reward/probability_outcome.py`+`reward/registry.py`
+`reward/protocol.py`、`verifier/`、`algorithms/dreamervla.py::world_model_pretrain_step`（WM warmup）、
`algorithms/registry.py`。

### 2.4 models（keep）
`world_model/wm_chunk.py`(`ChunkAwareWorldModel`)+deps（`wm.py`、`common.py`、`reward_heads.py`、
`block_linear.py`、`base_world_model.py`）、`reward/latent_success_classifier.py`、
`actor/latent_to_openvla_hidden_state_actor.py`(+`base_actor.py`、`_load.py`)、
`models/embodiment/openvla_oft/`、`models/encoder/`。

### 2.5 scripts（keep）
`e2e_coldstart_warmup_cotrain_ray.sh`、`e2e_coldstart_warmup_cotrain_noray.sh`、
`e2e_manual_cotrain_async.sh`、`eval_libero_vla.sh`、`run_wandb_relay_sync.sh`、
`collect_parallel.sh`，以及 env/download/preprocess 支撑脚本（`start_ray.sh`、`check_ray.sh`、
`install_env.sh`+`install/*`、`download_assets.sh`+`download/{20,30,40}_*.sh`、
`preprocess_libero.sh`+主线用到的 `preprocess/*`）。

### 2.6 configs 其它组（keep 主线成员）
`configs/dreamervla/openvla_onetraj_libero_cotrain_ray{,_base,_noray}.yaml`、
`configs/scripts/coldstart_warmup_cotrain.yaml`、`configs/scripts/eval_libero_vla.yaml`、
`configs/evaluation/libero_vla.yaml`、`configs/task/openvla_onetraj_coldstart_libero*.yaml`、
`configs/task/*_base_libero.yaml`（主线依赖）、`configs/logger/*`。

---

## 3. 废弃清单（→ archive/，激进）

> 迁移前对每条 `grep -rn` 确认主线 LUMOS/cotrain 路径不引用；确有引用则先解耦或保留。

### 3.1 configs/experiment（archive）
`collect_rollouts_ray_synthetic.yaml`、`online_cotrain_ray_dreamervla_tiny.yaml`、
`online_cotrain_ray_synthetic.yaml`、`latent_classifier_libero_goal_chunk{,_input_tokens}.yaml`、
`oft_discrete_token_world_model_chunk.yaml`、`oft_latent_classifier_chunk{,_input_tokens}.yaml`、
`oft_world_model_chunk{,_input_tokens}.yaml`、`world_model_chunk{,_input_tokens}.yaml`、
`world_model_step.yaml`、`openvla_oft_hdf5{,_one_trajectory,_one_trajectory_l1}.yaml`、
`vla_rynnvla_action_head.yaml`、`vla_sft_one_trajectory.yaml`。

### 3.2 runners（archive）
`backbone_dreamerv3_wm_runner.py`、`dreamerv3_pixel_runner.py`、`dreamerv3_token_runner.py`、
`chameleon_latent_action_wm_runner.py`、`latent_wm_runner.py`、`dreamervla_runner.py`、
`online_cotrain_ray_runner.py`（`OnlineCotrainRayRunner` 类本身；但**保留**
`..._ray_base.yaml`，主线在 `_ray.yaml` 中把 `_target_` 覆盖为 `ManualCotrainRayRunner`）、
`vla_sft_runner.py`、`openvla_oft_runner.py`。
辅助文件逐个实证再迁：`_dreamer_runner_common.py`、`frozen_wm_actor_critic.py`、
`collect_online_rollouts_for_classifier.py`、`collect_parallel_rollouts.py`（**先解 R3 collect
渲染入口疑点再定性**）、`vectorized_collect.py`、`vec_rollout_env.py`、
`rollout_hidden_extractor.py`、`rlinf_libero_rollout.py`、`pretokenize_vla_runner.py`、
`distributed.py`。

### 3.3 algorithms（archive）
`ppo/{dense,dense_chunk,grpo,relabel,tdmpc_critic}.py`、`algorithms/imagine/`、
`reward/sparse_outcome.py`，以及 `algorithms/dreamervla.py` 中的 dreamer actor-critic 部分
（`imagine_actor_critic_step`、`compute_lambda_returns`）——**注意** `world_model_pretrain_step`
在同文件且是主线，需按函数保留，不能整文件迁移。

### 3.4 models（archive）
world_model：`dreamer_v3_pixel_world_model.py`、`dreamer_v3_pixel_backbone_world_model.py`、
`dreamer_v3_token_world_model.py`、`dreamer_v3_token_from_pixel_world_model.py`、
`_dreamer_v3_token_common.py`、`dreamerv3_torch.py`、`tssm_backbone_world_model.py`、
`tssm_token_backbone_world_model.py`、`tssm_torch.py`、`chameleon_latent_action.py`。
actor：`latent_to_action_hidden_actor.py`、`latent_to_openvla_discrete_token_actor.py`、
`openvla_discrete_token_actor.py`、`rynnvla_action_hidden_actor.py`、`vla_action_head_actor.py`、
`vla_policy.py`。
critic：`models/critic/{critic,twohot_critic}.py`（仅 dreamer/LUMOS_DENSE 用；迁前 grep 确认
async LUMOS 主线不引用）。embodiment：`models/embodiment/chameleon_model/`、
`models/embodiment/openvla/`（部分已 archive）。

### 3.5 scripts / launchers / configs.scripts（archive）
`train_dreamervla.sh`、`train_vla.sh`、`train_wm.sh`、
`scripts/eval/launch_openvla_oft_official_libero_eval.sh`、`download/10_rynnvla.sh`、
`download/50_calvin_dataset.sh`、`preprocess/concat_record_libero.sh`、
`preprocess/30_action_hidden.sh`（及其它 rynn/action-hidden 专属 preprocess，逐个实证）；
`dreamervla/launchers/train.py`（若仅服务 standalone 训练则迁）、`launchers/workflow.py`（实证）；
`configs/scripts/{train_dreamervla,train_vla,train_wm,preprocess_rynn_pixel_hidden,`
`pretoken_state_action_model,concat_action_world_model_data_libero,openvla_oft_official_eval}.yaml`。

### 3.6 config 组 VLA/worldmodel/classifier
`configs/VLA/*`、`configs/worldmodel/*`、`configs/classifier/*` 仅服务 standalone
`world_model_*`/`oft_*`/`latent_classifier_*`/`vla_*` 实验（主线通过 `ray_components:` / task
`*_target` 内联构建 WM/classifier/actor，不走这些 override 组）——**逐文件确认无主线引用后迁移**。

---

## 4. eval encoder 隐患（先处理再废弃 rynn）

`configs/evaluation/libero_vla.yaml:36` 默认 encoder 为 `RynnVLAEncoder`；主线 eval 靠
`eval.ckpt_kind=dreamer` 覆盖走 dreamer encoder 路径。废弃 rynn encoder 前，必须确认
主线 eval（含 R1 的 `ckpt_kind=vla` base-VLA 路径与 `ckpt_kind=dreamer` 路径）**不硬依赖**
rynn 资产；若默认值指向 rynn，改默认为主线 encoder 或在 eval 配置中显式覆盖。

---

## 5. 分阶段执行计划（每步含 verify）

**Step 1 — 冻结主线契约（R2）**
- 锁定/同步双写点三值；在 `dreamervla/config.py` 加基线告警校验。
- verify：读校验分支；compose `openvla_onetraj_libero_cotrain_ray` 打印三值 = 32/256/512；
  tiny 覆盖时告警可见。

**Step 2 — EGL 三处对齐 · 单一 render helper（R3）**
- 2a 建 helper：扩展 `dreamervla/utils/egl_device.py` 实现
  `apply_libero_render_regime(backend, shard_id, gpu_pool)`（见 R3 统一方案）；配合
  `render_device_config.py` 的池校验。**先写无 GPU 单测**断言 egl/osmesa 下设对的 env var 集合
  与 `MUJOCO_EGL_DEVICE_ID` 选取逻辑（含零 GPU + egl 抛错）。
- 2b 实证 + 接线：确认主线 collect 真实渲染入口（`ColdStartRayCollectRunner` vs
  `collect_parallel_rollouts.py:497`），把 collect / cotrain-real / eval 三处 LIBERO env
  构造改成**只调 helper**（在各自 env-worker 子进程入口最早处、传该 worker shard id），
  删掉三处原有的各自 `os.environ.setdefault`/分歧 backend 逻辑。
- 2c 默认切换：三处 config 级 `render_backend` 默认改为 `egl`；保留 osmesa 显式回退与零 GPU 拒绝。
- verify：2a 单测绿（无 GPU 即可）；三处默认值 = egl；有 GPU 时 `render_backend=egl` 端到端
  冒烟不崩（尤其 collect/cotrain-real 不再 read_pixels SIGABRT）；无 GPU 标 GPU-GATED 并静态
  确认三处均只经 helper、每 worker 传对 shard id。

**Step 3 — base-VLA 基线 eval + 5 步双档验收（R1）**
- 先跑 `eval.ckpt_kind=vla` base 基线；再跑档 A（tiny smoke，`global_steps=5`）与
  档 B（真实 32/256/512，`global_steps=5`）。
- verify：base SR 与 cotrain SR 均入 `eval/` 命名空间并落盘；报告给出数值与上升趋势结论。

**Step 4 — 激进废弃（R4）**
- 逐类 `grep` 确认无主线引用 → `git mv` 到 archive/ → 写 manifest + 还原脚本 → 清理悬空 import。
- verify：主线 6 experiment 能 compose；单测（dreamervla env）绿；
  `restore_from_archive.sh --dry-run` 列全还原动作。

**Step 5 — 文档**
- 更新 `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`：写清 R1/R2/R3 的默认值、
  base-VLA 基线评测命令、EGL 默认、废弃与还原说明。
- verify：文档命令与实际 config key 一致（交叉核对）。

---

## 6. 回退机制（restore_from_archive.sh 契约）

- `--dry-run`：读 `docs/superpowers/DEPRECATION-manifest.md`，打印每条将执行的
  `git mv <archive_path> <orig_path>`，不改动。
- 无参 / `--all`：把 manifest 全部条目 `git mv` 回原位。
- `<path>...`：只还原指定文件。
- 幂等：目标已在原位则跳过并提示。
- 该脚本 + manifest 使"激进废弃"完全可回退，满足用户"必须可回退"要求。

---

## 7. 风险登记

1. **collect 渲染入口分歧**：映射对 `collect_parallel_rollouts.py` 是否在主线路径存疑 →
   Step 2 先实证再改/迁，避免改错文件。
2. **EGL 仍 SIGABRT**：回退 osmesa 并回 RLinf LIBERO `OffScreenRenderEnv` 对齐（已授权）。
3. **critic/imagine 误删**：迁前 grep 确认 async LUMOS 主线不引用（映射提示仅服务 dense 路由，需实证）。
4. **eval 默认 encoder = rynn**：§4 先处理，避免废弃 rynn 后 eval 崩。
5. **`algorithms/dreamervla.py` 混合**：主线 `world_model_pretrain_step` 与非主线
   dreamer actor-critic 同文件，按函数保留，禁止整文件迁移。
6. **在途未提交 diff**：用户有 per-rank/EGL 在途改动 → 只 `git add` 本次实际改动文件，
   切勿 `git add` 整个被预先修改过的文件。

---

## 8. 约束与护栏

- 单测/compose 用 dreamervla conda 环境（py3.11 / transformers 4.40.1）。
- 提交：conventional subject + `--signoff`；subject 不含 `===` 或 `/`；ruff 跑改动 py。
- 禁止改动含 "WoVR" 字样的措辞（active source 禁用词）。
- 废弃一律 `git mv`，绝不 `rm`。
- GPU 间歇可用：无 GPU 时做 CPU 可验证部分，需 GPU 的验证标 GPU-GATED，不算失败。
