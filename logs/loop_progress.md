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
| Step 2 | EGL 三处对齐（R3）：实证 collect 渲染入口，默认 collect/cotrain-real/eval 为 `egl`，复用/扩展 per-worker device binding，保留 osmesa 回退和零 GPU 拒绝。 | TODO | Verify: defaults are `egl`; GPU smoke if available, otherwise static path + GPU-GATED. |
| Step 3 | base-VLA 基线 eval + 5 步双档验收（R1）：base `eval.ckpt_kind=vla`，tiny 5 step，真实 32/256/512 5 step。 | TODO | Verify: base SR and cotrain SR under `eval/`, JSONL/TensorBoard persisted, trend reported. |
| Step 4 | 激进废弃（R4）：grep 主线引用，`git mv` 非主线到 `archive/`，更新 manifest 和 restore script，清理悬空 import。 | TODO | Verify: six mainline experiments compose, tests green, `restore_from_archive.sh --dry-run` lists restore actions. |
| Step 5 | 文档：更新 mainline tutorial，写清 R1/R2/R3 默认值、base-VLA 基线评测命令、EGL 默认、废弃与还原说明。 | TODO | Verify: doc commands and config keys cross-checked. |

## Current Atomic Step

- Step: `Step 1`
- Status: `DONE`
- Reason: R2 baseline values compose/read as 32/256/512, and overrides now emit visible `UserWarning` while remaining allowed for smoke/tiny.
