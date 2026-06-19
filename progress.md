# Progress Log

## Session 2026-06-19 — DinoWM query-stage worldmodel plan

### Done
- Created `docs/superpowers/plans/2026-06-19-dinowm-query-stage-worldmodel.md` as the active implementation plan for the current world-model work.
- Captured the required constraints: no multi-node default path, model/dataset decoupling, Hydra as the source of truth, query-before/query-after world-model taxonomy, DINO-WM concat conditioning, and VLA/dataset-derived dimensional contracts.
- Recorded the current architecture decisions, transformer sizing policy, parameter-count checkpoints, exact file touch points, implementation tasks, and verification matrix.

### Next
- Re-run the full focused unit/e2e/static verification matrix after the latest transformer-budget and plan updates.
- Update the OpenVLA one-trajectory tutorial once the current config and smoke-run path are re-verified.

## Session 2026-06-18

### Done
- Established context: prior WIP (`rlinf_libero_rollout.py` + planning files) was wiped by `git reset`.
- Read memory notes: RLinf parallel eval ~0.50 success_once on traj1; ray-backend effort paused.
- Confirmed `dreamervla` conda env has gym+libero+robosuite (native rollout possible).
- Confirmed docker container `rlinf` already running (use `docker exec` for RLinf eval).
- Dispatched 2 Explore agents → full RLinf reference contract + DreamerVLA current pipeline
  (see findings.md). Identified top 6 divergences.
- Created task_plan.md / findings.md / progress.md.

### Deliverable (this session = DOCUMENT ONLY, no execution per user)
- Wrote execution plan: `docs/experiment_tutorials/RLinf_aligned_LIBERO_rollout_execution_plan.md`
  (中文；策略=新写对齐 RLinf 的 rollout，不修补旧采集器；ray+非-ray；DoD/分阶段/对齐清单/风险)。

### Recon fact for Phase 0
- Running `rlinf` container NVML is BROKEN (`docker exec rlinf nvidia-smi` → Failed to init NVML).
  Phase 0 must `docker restart rlinf` or `docker run --gpus all` fresh. Host 8×H100 all idle.

### Phase 0 RESULT ✅ (2026-06-18, no redo per user)
- Fresh `docker run --gpus all` eval_wan_libero_goal_traj1_4567.sh (16 env, 1 epoch).
- RLinf/logs/20260618-15:17:54-*/: eval/success_once=0.50, success_at_end=0.375, 16 traj, len 512.
- Extra frozen consts: ACTION_PROPRIO_NORMALIZATION_TYPE=BOUNDS_Q99, PROPRIO_DIM=8 (unused), chunk=8, act_dim=7.

### Concurrency event
- A concurrent process churned the working tree (deleted+restored whole Ray backend, deleted my logs/ + draft doc).
  User said "现在恢复了" (restored). Ray kept as OPTIONAL backend per updated CLAUDE.md. Repo now clean (5 untracked).
- Execution plan doc rewritten clean at docs/experiment_tutorials/RLinf_aligned_LIBERO_rollout_execution_plan.md.

### ★★ ROOT CAUSE FOUND (Phase 2 blocker) — transformers version
- NEW core rlinf_libero_rollout.py gave 0/9 on libero_goal. So did ALL 28 canonical openvla-oft mp4s.
  => NOT my code; environmental.
- Host `dreamervla` env transformers=4.43.0; openvla-oft hard-requires 4.40.1 (modeling_prismatic.py:331);
  docker openvla-oft venv has 4.40.1 → RLinf 50%.
- Image render host==docker pixel-identical (ruled out). Golden A/B (same image+code+ckpt, only tf differs):
  4.40.1 → smooth coherent action chunk; 4.43.0 → garbage (max abs diff 1.59, gripper inverted). CONFIRMED.
- FIX = transformers==4.40.1. Already proven to work (golden + RLinf 50%). DECISION PENDING: how to apply
  (downgrade shared dreamervla env / dedicated env / docker) — affects shared env, asking user.
- Debug scratch in data/: _golden_action.py, _dbg_*.png, _act_*.npy (clean up after).

### ★★★ TRUE ROOT CAUSE = torch 2.5.1 vs 2.6.0 (precision/TF32/attn/transformers/timm all ruled out)
Layer-by-layer (identical saved inputs both envs): vision feat IDENTICAL; LLM input `inputs_embeds` IDENTICAL;
Llama OUTPUT differs (11.25), L00 onward, precision-independent (fp32 & TF32-off both give same host garbage).
Only env diff = torch 2.5.1(dreamervla) vs 2.6.0(docker/working).

### ★★★ FINAL ROOT CAUSE = vanilla transformers vs openvla-oft FORK
- openvla-oft needs the `moojink/transformers-openvla-oft` FORK (v4.40.1); host had VANILLA 4.40.1.
- Both report __version__ "4.40.1" (check passes) -> invisible. modeling_llama.py: vanilla 1566 vs fork 1620 lines.
- Swapped fork transformers into dvla_oft -> golden action flipped garbage [0.0003..] -> coherent [0.601..]==docker. ✓
- torch/timm/cuBLAS/weights/GPU all identical & ruled out; it was the fork all along.
- Saved memory: openvla-oft-needs-transformers-fork.md.

### ✅ Phase 2 DONE + env fixed in `dreamervla` + install scheme fixed
- `dreamervla` env: installed the FORK transformers (golden == docker [0.601..]). Backups:
  site-packages/transformers.vanilla443.bak + transformers-4.43.0.dist-info.bak (revert if needed).
- Rollout `dreamervla.runners.rlinf_libero_rollout` libero_goal task0,1,2 x3: **success_once=0.4444 (4/9)**
  (task0=0, task1=0.67, task2=0.67) — RLinf ballpark ~0.50. NON-ZERO confirmed.
- Install scheme fixed (per user): 30_python_deps.sh no longer pins transformers; 40_third_party.sh installs
  the fork with --force-reinstall + offline via TRANSFORMERS_OFT_FORK_SRC; 60_verify.sh asserts the fork is
  active (fails install on vanilla). SETUP.md §1 documents fork requirement + verify + offline. Offline fork
  source staged at data/_fork_transformers(+_distinfo).
- dvla_oft env still exists (stepping stone); can be removed since dreamervla now works.

### Phase 3/4 (in progress) — collectors: gripper post-process fix
- ROOT gap in ALL collector paths: no gripper post-process before env.step -> grasping fails.
- Added `process_action` to oft_collect_common.py (single source); rlinf_libero_rollout now imports it.
- Wired into: collect_parallel_rollouts.py (single-env, 2 sites), vectorized_collect.py (batched),
  rollout_inference_worker.py (ray; applied once there, NOT in env_worker -> no double-apply).
- chunk[0] (receding) is CORRECT for collectors (need per-step obs_embedding for WM; offline data is
  per-frame hidden) — only the gripper was missing. Recorded wm_action = post-processed (LIBERO scale, matches demos).
- Non-ray collector libero_goal task1,2: 2/4 success (task1 ep0, task2 ep1). ✅
- Ray collector libero_goal task1: 2/2 success (sparse_reward=1). ✅ env settles at reset (warmup_steps=10).
- Verified the OLD ray gripper fix (described in docs/superpowers eval-align plan §3) was LOST in churn; current
  ray runner files have no gripper handling -> my rollout_inference_worker process_action is the only one (no double-apply).

### ALL CORE PHASES DONE (0-4). Remaining optional: Phase 5 (object/spatial/10 suites); regression tests
### (gripper + rlinf_libero_rollout — were written by concurrent actor but lost in churn, not re-added).

### (historical) FIX path (b) — dedicated env dvla_oft
- dvla_oft = clone of dreamervla + FORK transformers (copied from docker /opt/venv).
- Snag: I had upgraded dvla_oft torch->2.6.0 (for the WRONG hypothesis); torch 2.6 defaults torch.load
  weights_only=True -> breaks LIBERO init_states load. Reverting dvla_oft torch -> 2.5.1 (clone original,
  libero-compatible). torch version is irrelevant to the fix (fork is the key).
- NEXT: re-verify golden + run rollout in dvla_oft (torch 2.5.1 + fork) -> expect non-zero success.
- Cloned dreamervla -> `dvla_oft` (has libero/gym/robosuite working, transformers 4.40.1, timm 0.9.10).
- Restored dreamervla to original (transformers 4.43.0, timm 0.9.16, torch 2.5.1). ✓ untouched.
- Upgrading dvla_oft torch -> 2.6.0+cu124 (torchvision 0.21.0). flash_attn not imported in inference path (sdpa).
- NEXT: golden test in dvla_oft (expect coherent [0.601,...]); then rollout non-zero.

### Phase 2 (was in progress) — NEW aligned action core written
- Wrote `dreamervla/runners/rlinf_libero_rollout.py`: shared RLinf-aligned core.
  Uses canonical `OpenVLAOFTObsActionPolicy` (get_vla_action) + `LIBERODreamerEnv` +
  process_action gripper (sign(2g-1)*-1) + full 8-step open-loop chunk + settle.
- GOTCHA found: `OpenVLAOFTObsActionPolicy.from_checkpoint` chdir's into openvla-oft root
  BEFORE resolving the ckpt path → MUST pass an ABSOLUTE ckpt path (diagnostics eval failed
  on a relative path). New module resolves to absolute before the call. ✓
- Running verification: libero_goal task 0,1,2 × 3 trials, gpu 0 → /tmp/dvla_phase2/aligned_rollout.log (be5d0s6pl).
  Target success_once ≥ ~0.4.

### Test results
| Run | Result |
|-----|--------|
| (none yet) | |
