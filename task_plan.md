# Task Plan — RLinf-aligned LIBERO rollout/eval/collector for OFT traj1

## Goal
Make DreamerVLA reproduce a **non-zero LIBERO success rate** with the OpenVLA-OFT
one-trajectory (discrete) checkpoint, matching the known-working RLinf eval
(~0.50 `success_once` on libero-goal). Deliver **both ray and non-ray** rollout
paths whose eval AND collector both show correct non-zero success.

Checkpoint root: `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-<suite>-traj1`
Primary suite: **libero-goal** (`unnorm_key=libero_goal_no_noops`).
Run env: conda `dreamervla` (has gym+libero+robosuite); `MUJOCO_GL=osmesa`.
Strategy (per user): **write NEW clean code that mirrors RLinf**, do NOT patch the
existing tangled collector/tutorial path.

## Definition of Done
- [ ] Phase 0 proves RLinf eval gives non-zero success on traj1 (recorded number).
- [ ] DreamerVLA standalone rollout (no collector) reaches success_once ≥ ~0.4 on libero-goal.
- [ ] Non-ray collector reaches the same non-zero success.
- [ ] Ray collector reaches the same non-zero success.
- [ ] eval + collector both verified non-zero with captured logs.

## Phases
| # | Phase | Status | Verify |
|---|-------|--------|--------|
| 0 | Prove RLinf eval non-zero (docker fresh container) | complete | DONE: success_once=0.50, success_at_end=0.375 (16 traj). No redo. |
| 1 | Freeze RLinf I/O reference contract → findings.md | complete | contract frozen (proprio=F, gripper binarize+invert, hist=1, chunk=8 open-loop, settle=15) |
| 2 | New RLinf-aligned rollout core + eval entry (no collector) | complete | DONE: success_once=0.4444 (4/9) libero-goal, dreamervla env + fork transformers |
| 3 | Non-ray collector on the aligned core | complete | DONE: 2/4 success (gripper fix), pushed a741e89 |
| 4 | Ray collector on the aligned core | complete | DONE: 2/2 success (sparse_reward=1), gripper fix in rollout_inference_worker (no double-apply; old fix lost in churn) |
| 5 | Generalize to other suites (object/spatial/10) | pending | per-suite non-zero (scope TBD) |

## Key Decisions
- Single shared "RLinf-aligned action step" core, reused by eval + non-ray + ray, so
  alignment lives in ONE place.
- Success bar = match RLinf ballpark (success_once ≥ ~0.4), not literal >0.
- libero-goal first, then generalize.

## Top Suspected Root Causes (from code diff, see findings.md)
1. Gripper post-process missing in DreamerVLA: RLinf does `g=2g-1` then `sign(g)*-1`.
2. Action-chunk execution: RLinf executes all 8 (open-loop); DreamerVLA executes only chunk[0].
3. Image frame count: RLinf 1 agentview frame (3ch); DreamerVLA may stack history=2 (6ch).
4. Initial 15 no-op settle steps (gripper -1) missing in DreamerVLA.
5. proprio usage parity for the discrete ckpt (verify RLinf use_proprio for traj1).
6. prompt trailing space `\nOut: ` vs `\nOut:`.

## ★ ROOT CAUSE (confirmed)
DreamerVLA rollout 0% (host `dreamervla` env) caused by **transformers 4.43.0**; openvla-oft requires
**4.40.1** (docker has it → RLinf 50%). Golden test proved 4.43 yields garbage discrete actions vs 4.40.1
coherent. ★★★ FINAL ROOT CAUSE = vanilla transformers vs openvla-oft FORK (moojink/transformers-openvla-oft v4.40.1).
Both report __version__ 4.40.1 (version check passes), but vanilla Llama forward produces garbage OFT actions;
fork's patched Llama is correct. Proof: modeling_llama.py vanilla 1566 lines vs fork 1620; swapping fork into
dvla_oft flipped golden action garbage->coherent==docker. NOT torch/timm/cuBLAS/weights/GPU (all identical).
FIX (option b): dedicated env `dvla_oft` (clone of dreamervla) with the FORK transformers (copied from docker).
dreamervla left untouched (restored to transformers 4.43.0 / timm 0.9.16).

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 0% success in dreamervla env (all tasks) | golden A/B vs docker | root cause = transformers 4.43 vs 4.40.1 |
| diagnostics eval OSError bad path | rel path after chdir | pass ABSOLUTE ckpt path (new module does) |
