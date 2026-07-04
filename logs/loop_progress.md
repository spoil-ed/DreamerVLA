# Mainline Deprecation EGL Align Loop Progress

## Sources

- Requested SPEC: `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`
- Mainline tutorial: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Current status: user corrected the authoritative fact-source path to the approved plan; the older copied specs path is no longer the loop source.

## Ledger

| ID | Step | Status | Notes |
| --- | --- | --- | --- |
| SPEC-0 | Locate and recover the authoritative SPEC at the requested path, then replace the provisional ledger with the SPEC step list. | DONE | Authoritative loop source is `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`; the older specs path is not used. |
| Step 1 | 冻结主线契约（R2）：锁定/sync `real=32`, `imagine=256`, `step=512` 双写点，并在 `dreamervla/config.py` 加基线告警校验。 | DONE | Verified current YAML double-write values are 32/256/512; added warning-only validation for baseline overrides. |
| Step 2-pre | EGL 三处对齐（R3）前置实证：确认主线 collect 真实渲染入口（`ColdStartRayCollectRunner` vs `collect_parallel_rollouts.py`）。 | DONE | Ray collect command uses `experiment=collect_rollouts_ray`; Hydra target is `ColdStartRayCollectRunner`; OFT path launches `WorkerGroup(EnvWorker, plan["env"])`. |
| Step 2a | EGL 三处对齐（R3）子步：扩展 `dreamervla/utils/egl_device.py`，新增 `apply_libero_render_regime(backend, shard_id, gpu_pool)` 并写无 GPU 单测。 | DONE | Added helper and no-GPU tests for egl/osmesa env vars, shard-indexed `MUJOCO_EGL_DEVICE_ID`, zero-GPU EGL rejection, and invalid backend rejection. |
| Step 2b | EGL 三处对齐（R3）子步：把 collect / cotrain-real / eval 三处 LIBERO env 构造改成只调 helper，且在各 worker 子进程入口最早处传 shard id。 | DONE | Wired collect EnvWorker, manual cotrain real TrajectoryEnvWorker, and post-step eval env through `apply_libero_render_regime()`; focused no-GPU tests pass. |
| Step 2c | EGL 三处对齐（R3）子步：三处 config 级 `render_backend` 默认改为 `egl`，保留 osmesa 显式回退与零 GPU 拒绝。 | DONE | Resolved collect EGL child death by defaulting collect render devices to GPUs disjoint from inference; 2-GPU real collect EGL smoke passed, single-GPU EGL fails fast with osmesa guidance, and direct cotrain-real EGL reset+step passed. |
| Step 3 | base-VLA 基线 eval + 5 步双档验收（R1）：base `eval.ckpt_kind=vla`，tiny 5 step，真实 32/256/512 5 step。 | TODO | Split into Step 3a/3b/3c so each iteration has one independently verifiable gate. |
| Step 3a | R1 子步：用 `EmbodiedEvalRunner` + `eval.ckpt_kind=vla` 跑原始 OpenVLA-OFT base-VLA 最小 LIBERO eval，并确认 SR 落盘到 run root。 | DONE | Added the OpenVLA-OFT base eval adapter in `EmbodiedEvalRunner`; one-task/one-episode EGL smoke wrote `eval_libero_metrics.json` with `eval_success_rate=0.0`. |
| Step 3b | R1 子步：`manual_cotrain_ray_tiny` 跑满 `manual_cotrain.global_steps=5`，确认 tiny 端到端 cotrain 绿。 | TODO | Verify: tiny run exits 0, resolved config has `global_steps: 5`, and metrics/log artifacts exist. |
| Step 3c | R1 子步：真实 32/256/512 配置跑满 5 global_step，并与 base-VLA SR 做趋势对比。 | TODO | Verify: real cotrain exits 0, eval namespace metrics persisted, report gives base/cotrain SR and trend. |
| Step 4 | 激进废弃（R4）：grep 主线引用，`git mv` 非主线到 `archive/`，更新 manifest 和 restore script，清理悬空 import。 | TODO | Verify: six mainline experiments compose, tests green, `restore_from_archive.sh --dry-run` lists restore actions. |
| Step 5 | 文档：更新 mainline tutorial，写清 R1/R2/R3 默认值、base-VLA 基线评测命令、EGL 默认、废弃与还原说明。 | TODO | Verify: doc commands and config keys cross-checked. |

## Current Atomic Step

- Step: `Step 3a`
- Status: `DONE`
- Reason: Base-VLA `eval.ckpt_kind=vla` now routes OpenVLA-OFT checkpoints through the EmbodiedEvalRunner harness and persists eval metrics. Step 3b is next.
