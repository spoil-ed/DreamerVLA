# Mainline Deprecation EGL Align Loop Progress

## Sources

- Requested SPEC: `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`
- Mainline tutorial: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Current status: SPEC recovered from the ignored approved plan at `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md` into the requested specs path.

## Ledger

| ID | Step | Status | Notes |
| --- | --- | --- | --- |
| SPEC-0 | Locate and recover the authoritative SPEC at the requested path, then replace the provisional ledger with the SPEC step list. | DONE | Recovered to `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`. |
| Step 1 | 冻结主线契约（R2）：锁定/sync `real=32`, `imagine=256`, `step=512` 双写点，并在 `dreamervla/config.py` 加基线告警校验。 | DONE | Verified current YAML double-write values are 32/256/512; added warning-only validation for baseline overrides. |
| Step 2a | EGL 三处对齐（R3）子步：实证主线 collect 真实渲染入口（`ColdStartRayCollectRunner` vs `collect_parallel_rollouts.py`）。 | DONE | Ray collect command uses `experiment=collect_rollouts_ray`; Hydra target is `ColdStartRayCollectRunner`; OFT path launches `WorkerGroup(EnvWorker, plan["env"])`. |
| Step 2b | EGL 三处对齐（R3）子步：默认 collect/cotrain-real/eval 为 `egl`，保留 osmesa 显式回退与零 GPU 拒绝。 | TODO | Next: declare/pass `env.cfg.render_backend` for Ray collect; direct `env.cfg.render_backend=egl` currently needs `+` because the key is absent. |
| Step 2c | EGL 三处对齐（R3）子步：复用/扩展 per-worker device binding for real env workers. | TODO | Verify: static device-binding path and GPU-GATED smoke as available. |
| Step 3 | base-VLA 基线 eval + 5 步双档验收（R1）：base `eval.ckpt_kind=vla`，tiny 5 step，真实 32/256/512 5 step。 | TODO | Verify: base SR and cotrain SR under `eval/`, JSONL/TensorBoard persisted, trend reported. |
| Step 4 | 激进废弃（R4）：grep 主线引用，`git mv` 非主线到 `archive/`，更新 manifest 和 restore script，清理悬空 import。 | TODO | Verify: six mainline experiments compose, tests green, `restore_from_archive.sh --dry-run` lists restore actions. |
| Step 5 | 文档：更新 mainline tutorial，写清 R1/R2/R3 默认值、base-VLA 基线评测命令、EGL 默认、废弃与还原说明。 | TODO | Verify: doc commands and config keys cross-checked. |

## Current Atomic Step

- Step: `Step 2a`
- Status: `DONE`
- Reason: Mainline Ray collect render path is proven as launcher `collect_rollouts_ray` -> `ColdStartRayCollectRunner` -> `EnvWorker` env cfg, not `collect_parallel_rollouts.collect_rollouts()`.
