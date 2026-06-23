# 待办计划集中区 — INDEX(已合并)

整理日期:2026-06-19(本次清理:删除已完成计划,把"未确认完成"的计划合并进本文件)。

> 本目录原先散着十来份 plan/TODO。多数工作其实**已在仓库且验证过**,只是 churn 后没回填勾选。
> 2026-06-19 做了一次合并:**完全完成的删掉**,**未确认完成的合并到这里**(下面的"剩余工作"段)。
> 现在 `docs/superpowers/TODO/` 只剩本 INDEX;`docs/superpowers/specs/` 保留为设计存档;
> `docs/superpowers/plans/` 已空。

## 已删除(完全完成,代码在仓且验证过)

| 原文件 | 完成情况 |
|---|---|
| `coldstart_vs_rlinf_eval_layer_diff_plan.md` | 首处分歧根因 = vanilla vs transformers fork,已修复+测试;§5 逐层 diff harness 作废 |
| `2026-06-16-parallel-rollout-collector.md` | 非-ray 采集器已在仓且验证 |
| `2026-06-17-coldstart-collector-hydra.md` | 纯-Hydra 冷启动采集链路已工作(教程 `OpenVLA_Onetraj_LIBERO_coldstart_rollout_collection.md`) |
| `2026-06-18-ray-coldstart-real-oft-wiring.md` | ray 冷启动采集器验证 2/2 成功 |
| `2026-06-11-package-python-cli-migration.md` | `scripts/` 无 `.py`,shell 入口走 `python -m dreamervla...`;有 hygiene 测试 |

## 已合并到本文件并删除(未确认完成)

`2026-06-17-ray-backend-implementation-progress.md`(ray scaffolding 进度/handoff)、
`2026-06-18-ray-coldstart-review-fixes.md`(ray 冷启动 review-fix)、
`2026-06-17-offline-warmup-online-cotrain-pipeline.md`(离线 warmup→在线 cotrain 流水线)。
它们的设计主体均已实现;**仅剩的真实工作**已抽到下面"剩余工作"。

## 已作废(非目标)

- `plans/2026-06-19-vram-autosize.md`(VRAM 预算自动反算 micro-batch / env 数)——
  已删。本仓改走 **RLinf 手动挡哲学**,VRAM 自适应判为非目标,理由见
  `docs/ray_rlinf_alignment_implemented.md` §3.1。

---

## 剩余工作(合并后的真实待办,按优先级)

> **Ray backend 的待办已迁出本文件**,单一来源是 **`docs/ray_rlinf_alignment_todo.md`**(P0/P1 已完成,
> 剩余为条件型 P3;多节点已明确为非目标);已实现部分见
> **`docs/ray_rlinf_alignment_implemented.md`**。下面只留**非 ray-backend、本目录独有**的待办。

1. **offline-warmup → online-cotrain 真实长跑验证**:
   `OnlineCotrainPipelineRunner`、offline seed、warmup split ckpt、config/launcher 与轻量测试均已具备;
   仍缺**真实 OFT/LIBERO 端到端长跑**确认收敛/指标。

2. **(可选)OFT 主任务 Phase 5**(记于根 `task_plan.md`):
   把已验证的 libero-goal 方案泛化到 object / spatial / 10 三套件(同一 transformers fork + 夹爪修复
   应通用,未逐套件跑)。

---

## 根目录文件(保留,勿删)

`task_plan.md` / `progress.md` / `findings.md` 仍在仓库根:① 它们被 `planning-with-files` 钩子按**根路径**引用;
② 它们是 git **tracked**(删除是版本控制改动);③ `task_plan.md` 的 Phase 5 仍是 open。本次不动。
