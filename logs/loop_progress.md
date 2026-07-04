# Mainline Deprecation EGL Align Loop Progress

## Sources

- Requested SPEC: `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`
- Mainline tutorial: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Current status: user corrected the authoritative fact-source path to the approved plan; the older copied specs path is no longer the loop source.

## Ledger

| ID | Step | Status | Notes |
| --- | --- | --- | --- |
| SPEC-0 | Locate and recover the authoritative SPEC at the requested path, then replace the provisional ledger with the SPEC step list. | DONE | Recovered to `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`. |
| Step 1 | 冻结主线契约（R2）：锁定/sync `real=32`, `imagine=256`, `step=512` 双写点，并在 `dreamervla/config.py` 加基线告警校验。 | DONE | Verified current YAML double-write values are 32/256/512; added warning-only validation for baseline overrides. |
| Step 2-pre | EGL 三处对齐（R3）前置实证：确认主线 collect 真实渲染入口（`ColdStartRayCollectRunner` vs `collect_parallel_rollouts.py`）。 | DONE | Ray collect command uses `experiment=collect_rollouts_ray`; Hydra target is `ColdStartRayCollectRunner`; OFT path launches `WorkerGroup(EnvWorker, plan["env"])`. |
| Step 2a | EGL 三处对齐（R3）子步：扩展 `dreamervla/utils/egl_device.py`，新增 `apply_libero_render_regime(backend, shard_id, gpu_pool)` 并写无 GPU 单测。 | DONE | Added helper and no-GPU tests for egl/osmesa env vars, shard-indexed `MUJOCO_EGL_DEVICE_ID`, zero-GPU EGL rejection, and invalid backend rejection. |
| Step 2b | EGL 三处对齐（R3）子步：把 collect / cotrain-real / eval 三处 LIBERO env 构造改成只调 helper，且在各 worker 子进程入口最早处传 shard id。 | DONE | Wired collect EnvWorker, manual cotrain real TrajectoryEnvWorker, and post-step eval env through `apply_libero_render_regime()`; focused no-GPU tests pass. |
| Step 2c | EGL 三处对齐（R3）子步：三处 config 级 `render_backend` 默认改为 `egl`，保留 osmesa 显式回退与零 GPU 拒绝。 | BLOCKED | Defaults/tests are wired and collect now injects an EGL render pool, but real Ray collect EGL smoke still loses the EnvWorker child during step; explicit osmesa fallback passes. |
| Step 3 | base-VLA 基线 eval + 5 步双档验收（R1）：base `eval.ckpt_kind=vla`，tiny 5 step，真实 32/256/512 5 step。 | TODO | Verify: base SR and cotrain SR under `eval/`, JSONL/TensorBoard persisted, trend reported. |
| Step 4 | 激进废弃（R4）：grep 主线引用，`git mv` 非主线到 `archive/`，更新 manifest 和 restore script，清理悬空 import。 | TODO | Verify: six mainline experiments compose, tests green, `restore_from_archive.sh --dry-run` lists restore actions. |
| Step 5 | 文档：更新 mainline tutorial，写清 R1/R2/R3 默认值、base-VLA 基线评测命令、EGL 默认、废弃与还原说明。 | TODO | Verify: doc commands and config keys cross-checked. |

## Current Atomic Step

- Step: `Step 2c`
- Status: `BLOCKED`
- Reason: GPU collect EGL smoke reaches policy/env startup but the spawned EnvWorker child dies during step; next iteration should align the Ray spawn LIBERO OffScreenRenderEnv path with RLinf or choose the documented fallback.
