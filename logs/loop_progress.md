# Mainline Deprecation EGL Align Loop Progress

## Sources

- Requested SPEC: `docs/superpowers/specs/2026-07-04-mainline-deprecation-egl-align-design.md`
- Mainline tutorial: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Current status: SPEC is missing from the current worktree, git refs, and attachment search. This ledger is provisional and uses only the explicit R1-R4 acceptance criteria from the pasted objective until the SPEC is restored.

## Ledger

| ID | Step | Status | Notes |
| --- | --- | --- | --- |
| SPEC-0 | Locate and read the authoritative SPEC, then replace this provisional ledger with the SPEC step list. | BLOCKED | The named SPEC path does not exist; `git log --all`, `git rev-list --all --objects`, workspace search, and attachment search found no copy. |
| R1 | Baseline VLA SR through `eval.ckpt_kind=vla`, then cotrain runs 5 `global_step` with improving SR for both `manual_cotrain_ray_tiny` and real `32real/256imagine/global_steps=5`. | TODO | Provisional item from pasted objective; exact SPEC substeps unknown. |
| R2 | Lock `real=32`, `imagine=256`, `step=512` in Ray and launcher configs, with `dreamervla/config.py` early validation. | TODO | Provisional item from pasted objective; exact SPEC substeps unknown. |
| R3 | Align default EGL for collect/cotrain-real/eval, per-worker EGL device assignment, OSMesa fallback, and zero-GPU EGL rejection. | TODO | Provisional item from pasted objective; exact SPEC substeps unknown. |
| R4 | Move all non-mainline routes/files to `archive/` with manifest and restore script while keeping mainline six experiments composable and tests green. | TODO | Provisional item from pasted objective; exact SPEC substeps unknown. |

## Current Atomic Step

- Step: `SPEC-0`
- Status: `BLOCKED`
- Reason: The loop requires reading the authoritative SPEC before selecting and implementing a real TODO. That file is absent in the current local state.
