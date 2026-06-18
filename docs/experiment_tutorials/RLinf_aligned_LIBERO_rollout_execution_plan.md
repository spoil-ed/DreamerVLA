# 执行计划：把 RLinf 的 OFT-traj1 LIBERO 方案迁移到 DreamerVLA

> 目的：让 DreamerVLA 用 OpenVLA-OFT 单轨迹（discrete）权重在 LIBERO 上跑出
> **非零成功率**，对齐已验证可用的 RLinf eval；交付 **eval / 非-ray 采集器 / ray 采集器**
> 三条路径，且**共用同一套"RLinf 对齐动作核心"**。
>
> 策略（按用户要求）：**新写一份干净、直接对齐 RLinf 的动作核心代码**，三条路径都调用它；
> 不在各处复制动作/图像/prompt/夹爪/chunk 逻辑。

---

## 1. 目标与完成定义

**权重**：`data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-<suite>-traj1`
**主攻套件**：`libero_goal`（`unnorm_key=libero_goal_no_noops`），跑通再泛化。
**运行环境**：host 原生 conda `dreamervla`（已装 `gym+libero+robosuite`）；`MUJOCO_GL=osmesa`。

**完成定义（Definition of Done）**
1. ✅ Phase 0：RLinf eval 在 traj1 上非零成功（**已完成，见 §5**）。
2. DreamerVLA standalone rollout（不接采集器）在 libero-goal 上 `success_once ≥ ~0.4`。
3. 非-ray 采集器复用同一动作核心，采集过程统计到非零 `success_once`。
4. ray 采集器复用同一动作核心，达到同量级非零成功率。
5. eval 与采集器两侧都有**已捕获日志**佐证非零成功。

> 成功率门槛取"与 RLinf 同量级（`success_once ≥ ~0.4`，RLinf=0.50）"，而非字面 `>0`。

---

## ★ 实际根因与修复（2026-06-18 确认）

> 0% 的真正原因**不是**动作对齐层，而是 **`dreamervla` 环境的 transformers 版本错配**。
>
> - openvla-oft 在 `third_party/openvla-oft/prismatic/extern/hf/modeling_prismatic.py:331`
>   **硬性要求 `transformers==4.40.1` + `tokenizers==0.19.1`**；docker 用的正是 4.40.1（→RLinf 50%），
>   host `dreamervla` 却是 **4.43.0** → discrete `predict_action`/`generate` 输出**垃圾动作** → 全任务 0%。
> - Golden 对照（同图同代码同权重，仅 transformers 不同）：4.40.1 出平滑连贯动作块；
>   4.43.0 出乱跳噪声（max abs diff 1.59，夹爪维近乎相反）。图像/env 渲染 host==docker 像素一致，已排除。
> - **修复**：`pip install transformers==4.40.1`（tokenizers 0.19.1 本就正确）。
>   回滚参考：原为 `transformers==4.43.0`。
> - 我的对齐核心 `dreamervla/runners/rlinf_libero_rollout.py` 逻辑正确，仅被错的 transformers 拖垮。

## 2. 运行环境与资产（已侦察确认）

- **DreamerVLA 侧（host 原生）**：conda `dreamervla` 已装 `gym+libero+robosuite`，
  rollout/采集可直接在 host 跑；渲染 `export MUJOCO_GL=osmesa`（本机 EGL 在
  robosuite `read_pixels` 崩）。4 套 traj1 权重均在 `data/checkpoints/Openvla-oft-SFT-traj1/`。
- **RLinf 参考侧（docker，仅作 oracle）**：镜像
  `rlinf/rlinf:agentic-rlinf0.2-maniskill_libero`。
  ⚠️ 已运行容器 `rlinf` 的 NVML 损坏；需 `docker run --rm --gpus all` 起新容器。
  host `/mnt/data/spoil/workspace` → 容器 `/workspace/RLinf`。
  ⚠️ launcher 默认 MODEL_PATH 指向过期 `data/ckpts/...`，必须覆盖为 `data/checkpoints/...`。
  入口 `examples/embodiment/eval_embodied_agent.py`，配置 `wan_libero_goal_grpo_openvlaoft_4567`。

---

## 3. RLinf 参考 I/O 契约（已冻结 — 迁移真值来源）

来自实测 resolved-config dump + RLinf 源码 + eval 日志常量，逐项确定：

| 环节 | 值 / 行为 |
|---|---|
| 相机 | 256×256；env 出 agentview + eye_in_hand，但**只喂 1 张 agentview** |
| 图像帧数 | `num_images_in_input=1`，**单帧**，无 wrist、无时间堆叠 |
| 旋转 | env 级 180°：`img[::-1, ::-1]`（只转一次） |
| 预处理 | resize→224 + `center_crop=true` + `/255` + ImageNet 归一化；RGB；`image_size=[224,224]` |
| proprio | **`use_proprio=false`，不喂 proprio**（discrete 前向只走 vision feature） |
| prompt | `f"In: What action should the robot take to {task.lower()}?\nOut: "`（**"Out:" 后有尾空格**）；`max_prompt_length=128` |
| 解码 | discrete：`argmax → vocab_size - id → clip → bin_centers → _unnormalize_actions` |
| 反归一化 | `ACTION_PROPRIO_NORMALIZATION_TYPE = BOUNDS_Q99`（用 **q01/q99**，非 min/max） |
| 动作块 | `num_action_chunks=8`，`action_dim=7` |
| 夹爪 | 进后处理前 ∈ `[0,1]`；`g=2g-1` → `g=sign(g)*-1.0`（**二值化@0.5 + 反向** → ±1.0） |
| 动作执行 | **整块 8 步开环执行**完才重查策略（`libero_env.py chunk_step` `for i in range(8)`） |
| 初始静置 | reset 后 **15 步 no-op**，夹爪保持 `-1`（`reset_gripper_open`） |
| 模型 | `from_pretrained` 本地目录，**bf16**，`flash_attention_2`；`dataset_statistics.json` 取自 model_path |
| unnorm_key | `libero_goal_no_noops`（按套件切换） |
| 成功指标 | `success_once`=回合内曾 terminated；`success_at_end`=末步 terminated |
| eval 常量 | `max_episode_steps=512`，`ignore_terminations=true`，`auto_reset=true`，`seed=0` |

---

## 4. DreamerVLA 现状差异（按"最可能导致 0%"排序）

| # | 差异 | RLinf（对） | DreamerVLA 现状（疑似错） |
|---|---|---|---|
| 1 | 夹爪后处理 | `2g-1` 再 `sign*-1` 二值化 | 未见该步，原样下发 → 抓取必败 |
| 2 | 动作块执行 | 整块 8 步开环 | 只执行 `chunk[0]`（receding） |
| 3 | 图像帧数 | 单帧(3ch) | 某些路径 history=2 堆 2 帧(6ch) |
| 4 | 初始静置 | 15 步 no-op | 未见 |
| 5 | prompt 尾空格 | `\nOut: ` | `\nOut:` |
| 6 | 图像算子 | torchvision resize+crop+ImageNet | TF lanczos3+crop_and_resize（数值接近，低风险） |

> proprio 两侧一致（都不喂），不是差异点。

---

## 5. 分阶段执行计划

### Phase 0 — 证明 RLinf eval 非零  ✅ 已完成（2026-06-18，无需重做）
新容器 `docker run --gpus all` 跑 `eval_wan_libero_goal_traj1_4567.sh`
（`TOTAL_NUM_ENVS=16 EVAL_ROLLOUT_EPOCH=1`，MODEL_PATH 覆盖到 `data/checkpoints/...`）。
**结果**（`RLinf/logs/20260618-15:17:54-*/`）：

```
eval/success_once   = 0.50
eval/success_at_end = 0.375
eval/num_trajectories = 16   episode_len = 512
```

### Phase 1 — 冻结参考契约  ✅ 已完成
§3 全部敲定（proprio=false、gripper 二值化+反向、history=1、BOUNDS_Q99、chunk=8 开环、settle=15）。
可选未做项：抓一帧 RLinf golden `pixel_values`/首个 action chunk 数值做数值对照（非阻塞）。

### Phase 2 — 新写 RLinf 对齐动作核心 + standalone eval（不接采集器）
**目标**：DoD#2 —— libero-goal `success_once ≥ ~0.4`，host `dreamervla` 原生跑。
- 新建对齐核心（命名示意 `dreamervla/runners/rlinf_libero_rollout.py`），严格实现 §3：
  单帧 agentview、180° 旋转、resize+center_crop+ImageNet norm、prompt（含尾空格）、
  discrete 解码、`BOUNDS_Q99` 反归一化、**夹爪 `2g-1`+`sign*-1`**、**整块 8 步开环执行**、
  **15 步初始静置**、不喂 proprio。
- 配最小 eval 入口：N 个 LIBERO env、512 步上限、统计 `success_once/at_end`。
- 复用现有正确原语（模型加载、bin_centers 解码）即可，只把 §4 的差异处按 RLinf 实现。
- **命令骨架**：
  ```bash
  conda activate dreamervla && export MUJOCO_GL=osmesa
  CUDA_VISIBLE_DEVICES=0 python -m dreamervla.runners.rlinf_libero_rollout \
    model_path=data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
    unnorm_key=libero_goal_no_noops num_envs=16 max_steps=512
  ```
- **验证**：`success_once ≥ ~0.4`。未达标按 §6 清单逐项二分。

### Phase 3 — 非-ray 采集器（复用对齐核心）
让非-ray 采集路径调用 Phase 2 的核心产生动作，外层只管并行 env 与按既有 HDF5+sidecar 落盘。
小规模采集统计 `success_once>0`，且 sidecar 校验通过、下游消费契约不变。

### Phase 4 — ray 采集器（复用对齐核心）
ray worker 用**同一核心**做批量动作推理（可做 ray vs 非-ray 同 seed 逐步动作一致性对照）。
小规模 ray 采集统计 `success_once>0`，与非-ray 同量级。

### Phase 5 —（可选）泛化到 object / spatial / libero_10
换 ckpt + unnorm_key 复跑 eval + 采集，确认非零。

---

## 6. 对齐核对清单（实现/排障必过）

- [ ] 单帧 agentview（不堆历史帧），RGB。
- [ ] 180° 旋转一次（`img[::-1,::-1]`）。
- [ ] resize 224 + center_crop + `/255` + ImageNet norm。
- [ ] prompt `...?\nOut: `（尾空格），`max_prompt_length=128`。
- [ ] 不喂 proprio。
- [ ] 解码 `argmax→vocab_size-id→clip→bin_centers→_unnormalize(BOUNDS_Q99)`。
- [ ] unnorm_key 与套件匹配；`dataset_statistics.json` 已加载。
- [ ] 夹爪 `g=2g-1` 再 `g=sign(g)*-1`（二值化+反向）。
- [ ] **整块 8 步开环执行**（不要只执行 chunk[0]）。
- [ ] reset 后 15 步 no-op，夹爪 -1。
- [ ] 模型 bf16；动作 float32；回合上限 512。

---

## 7. 验证方法与风险

- 主指标 `success_once`，辅 `success_at_end`；每步把数字/命令写进 `progress.md`，失败写
  `task_plan.md` Errors 表。
- Phase 2 起建议先做"golden 数值对照"（`pixel_values`/首 action chunk 与 RLinf 一致），
  再看成功率，避免成功率为 0 时无从下手。
- 风险：① host(`dreamervla`) 与 docker(RLinf) 的 libero/robosuite 版本差异 → 必要时在 docker 内
  交叉验证；② 同一报错最多 3 次、每次换法，3 次不过则暂停上报。
- 日志/产物写到**仓库外或固定目录**，避免被 git 清理误删。

---

## 8. 文件索引

- 参考（RLinf）：`examples/embodiment/eval_wan_libero_goal_traj1_4567.sh`、
  `rlinf/envs/libero/libero_env.py`、`rlinf/envs/action_utils.py`、
  `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`。
- 本仓库（DreamerVLA）：`dreamervla/envs/libero_env.py`、
  `dreamervla/runners/rollout_hidden_extractor.py`、`dreamervla/runners/oft_collect_common.py`、
  `dreamervla/runners/collect_parallel_rollouts.py`、
  `dreamervla/runners/cold_start_ray_collect_runner.py`、
  `dreamervla/workers/inference/rollout_inference_worker.py`。
- 进度：根目录 `task_plan.md` / `findings.md` / `progress.md`。
