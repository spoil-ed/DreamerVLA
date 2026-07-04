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
| Step 2b | EGL 三处对齐（R3）子步：把 collect / cotrain-real / eval 三处 LIBERO env 构造改成只调 helper，且在各 worker 子进程入口最早处传 shard id。 | DONE | Wired collect EnvWorker, manual cotrain real TrajectoryEnvWorker, post-step/standalone eval, and sync no-Ray cotrain env construction through `apply_libero_render_regime()`; focused no-GPU tests pass. |
| Step 2c | EGL 三处对齐（R3）子步：三处 config 级 `render_backend` 默认改为 `egl`，保留 osmesa 显式回退与零 GPU 拒绝。 | DONE | Resolved collect EGL child death by defaulting collect render devices to GPUs disjoint from inference; standalone eval now defaults `eval.render_backend=egl` with `eval.render_gpu_pool` override; 2-GPU real collect EGL smoke passed, single-GPU EGL fails fast with osmesa guidance, and direct cotrain-real EGL reset+step passed. |
| Step 3 | base-VLA 基线 eval + 5 步双档验收（R1）：base `eval.ckpt_kind=vla`，tiny 5 step，真实 32/256/512 5 step。 | TODO | Split into Step 3a/3b/3c so each iteration has one independently verifiable gate. |
| Step 3a | R1 子步：用 `EmbodiedEvalRunner` + `eval.ckpt_kind=vla` 跑原始 OpenVLA-OFT base-VLA 最小 LIBERO eval，并确认 SR 落盘到 run root。 | DONE | Added the OpenVLA-OFT base eval adapter in `EmbodiedEvalRunner`; one-task/one-episode EGL smoke wrote `eval_libero_metrics.json` with `eval_success_rate=0.0`. |
| Step 3b | R1 子步：`manual_cotrain_ray_tiny` 跑满 `manual_cotrain.global_steps=5`，确认 tiny 端到端 cotrain 绿。 | DONE | CPU/Ray tiny path ran with `manual_cotrain.global_steps=5` and `manual_cotrain.learner_update_step=1`; run root contains resolved config, manifest, and finished progress JSON for global_step 1-5. |
| Step 3c | R1 子步：真实 32/256/512 配置跑满 5 global_step，并与 base-VLA SR 做趋势对比。 | BLOCKED | `/tmp/dvla-step3c-real-eval-gate-v1/cotrain` completed 5 EGL global steps and wrote `manual_cotrain_step_5`; cotrain/Ray logs have no `read_pixels`/SIGABRT/OOM. In-run eval failed before SR: first on Dreamer+OFT eval adapter (`self.encoder is None`, fixed with a unit test), then on LIBERO task initialization with process abort 134, so cotrain SR/trend is still unavailable. |
| Step 4a | R4 回退基础设施：manifest 驱动的 `restore_from_archive.sh` + 单测。 | DONE | commit `2baf4b7`. `restore_from_archive.sh` 以 `DEPRECATION-manifest.md` 为源，dry-run 列全 74 还原动作；3 单测绿。manifest 本身由并发会话 `30aab68` 先建(74 行中文版，等价），本轮误重写后已恢复原版。仅提交脚本+单测，74 staged rename 未卷入。 |
| Step 4b | R4 逐批归档：`git mv` SPEC §3 尚在原位的 ~80 非主线文件(experiment/runners/algorithms/models/configs 组)，每批 grep 主线无引用 + 追加 manifest + compose。 | TODO | 串行执行，避免 manifest/index 冲突。algorithms/dreamervla.py 按函数拆(world_model_pretrain_step 主线保留)。 |
| Step 5 | 文档：更新 mainline tutorial，写清 R1/R2/R3 默认值、base-VLA 基线评测命令、EGL 默认、废弃与还原说明。 | TODO | Verify: doc commands and config keys cross-checked. |

## Current Atomic Step

- Step: `R3/R1 — clean base landed; NEXT = implement RLinf-aligned subprocess eval fix`
- Status: `UNBLOCKED — in-flight state committed (2249054)`
- Done this step: user authorized committing the whole in-flight state. Committed `2249054`
  (61 files: in-flight EGL/eval/per-rank refactor + classifier_metrics decoupling + restore-script
  setup-scripts curation), 74 archive renames kept staged separate. Closed-loop: my restore-script
  (2baf4b7) had broken 2 test_setup_scripts assertions (uncurated/unregistered script); fixed by
  adding it to the curated top-level set + allowed_unregistered (the release-docs word ban forbids
  "archive" in scripts/README.md, so it is manifest-documented, not README-registered). Committed
  tree composes 6/6. Known remaining failures (documented, NOT mine): test_env_full_record,
  test_learner_worker_manual_precision, test_multistep_rollout_worker (in-flight WIP),
  test_repository_hygiene (loop reports under logs/).
- NEXT (eval fix): route the dreamer eval `_evaluate_libero_online_latent` LIBERO env through the
  SAME subprocess mechanism collect/cotrain use (VecRolloutEnv / env_worker + apply_libero_render_regime)
  instead of in-process get_libero_env, so mujoco EGL is isolated from the eval torch-CUDA context.
  Acceptance: ISOMORPHIC to collect/cotrain (same class+helper) + max_steps=300 eval SUSTAINS
  (no abort 134). GPU-gated verify.
- Original blocker cleared: eval files were entangled with the in-flight refactor (515 lines); now
  committed, so the fix can be implemented on a clean base without a merge conflict.
- Decisions (user 2026-07-04): eval fix = align to RLinf (NOT osmesa); R4 = aggressive
  (repoint train.yaml default + rewrite route tests). Acceptance add-on: eval must be
  ISOMORPHIC to collect/cotrain (same subprocess class + same `apply_libero_render_regime`),
  and a max_steps=300 eval must SUSTAIN.
- RLinf review (done, `/mnt/data/spoil/workspace/RLinf`): LIBERO renders in SUBPROCESS
  (`envs/libero/venv.py` SubprocVectorEnv) isolated from torch; EGL device via
  `scheduler/hardware/accelerators/nvidia_gpu.py:114`; `reconfigure` closes+recreates env
  between episodes. Fix = route eval LIBERO env through DreamerVLA's existing
  VecRolloutEnv/env_worker subprocess (same helper) instead of in-process `get_libero_env`.
- BLOCKER: the 3 eval files (`embodied_eval_runner.py` +468, `_embodied_eval_latent_mixin.py`
  +93, `pretokenize_vla_runner.py` +76 = 515 in-flight uncommitted lines) are being actively
  refactored (orthogonal base-VLA adapter work, no subprocess isolation). Implementing the fix
  now = 515-line conflict. Need the in-flight eval diff committed (or explicit go-ahead to build
  on it) before implementing. See egl-eval-fix-rlinf-subprocess-plan memory.
- Empirical crash judgment (current code, prior step): collect/cotrain SUSTAIN (subprocess);
  eval max_steps=50 OK, 300 CRASH 2/2 (in-process mjr_readPixels abort, empty_cache ruled out).
- Reason: Fresh current-code empirical test (single GPU pinned, render_backend=egl, dreamer
  ckpt manual_cotrain_step_5, libero_goal task 0, 1 episode). Result table:
  max_steps=50 → SUSTAIN (rc=0, eval_libero_metrics.json written, 0 aborts);
  max_steps=300 → CRASH, DETERMINISTIC (2/2 runs on GPU0 and GPU3, abort 134 / core dumped,
  no metrics). Crash is cumulative within a single episode (leak between 50 and 300 in-process
  mjr_readPixels calls) and driver-level (gdb: abort ← libnvidia-eglcore ← mjr_readPixels ←
  mujoco/_render). Wiring is NOT the cause: two Explore traces confirm collect + cotrain-real +
  all 3 eval entries genuinely resolve to egl and unify on apply_libero_render_regime.
  ASYMMETRY that explains sustain-vs-crash: collect/cotrain render in SPAWN SUBPROCESSES with
  native-crash catch + respawn (env_worker.py:535, EOFError/OSError → respawn), so they survive;
  EVAL renders IN-PROCESS (get_libero_env inside the runner) with NO isolation, so the first
  readPixels SIGABRT kills the whole eval. JUDGMENT: collect/cotrain-real EGL = sustainable;
  eval EGL = NOT sustainable for real-length episodes (needs max_steps~300, 10 tasks) as-is.
- Next: fix eval EGL crash. Preferred = give eval the same subprocess render isolation collect has
  (or fix the per-step EGL resource leak); osmesa fallback for eval only is the SPEC-sanctioned
  last resort. NB: eval runtime files overlap the user's in-flight uncommitted diff — coordinate.
