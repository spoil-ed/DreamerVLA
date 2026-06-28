# HISTORY — what's been built (done-work master outline)

> Summarized log of shipped work — the "what already exists" outline. The full per-commit
> detail lives in `git log`; this is the code-level summary that used to be scattered across
> `docs/history/`. Architecture & rules → [AGENTS.md](../AGENTS.md); open work → [TODO.md](superpowers/TODO.md).
> When an item lands, add a line here (with its commit) and remove it from `TODO.md`.

- Last updated: 2026-06-27

---

## Training entry & routes
- Single grouped Hydra entry: `python -m dreamervla.train experiment=<name> task=<suite>` →
  `Runner.setup()/execute()/teardown()`. Public runners exported from `dreamervla.runners`;
  route YAMLs `_target_` the public names. Config groups: `experiment/ VLA/ worldmodel/
  classifier/ dreamervla/ evaluation/ task/ logger/`. Thin shell launchers (`train_vla.sh`,
  `train_wm.sh`, `train_dreamervla.sh`, `eval_libero_vla.sh`) forward `key=value`.
- Active routes: VLA SFT (rynnvla / openvla_oft, full + one-trajectory), world-model
  (`*_dinowm_chunk`, input-token, discrete-token), classifiers, three `dreamervla_*` recipes
  (actor_critic, rynn/oft LUMOS), `eval_libero_vla`.

## Ray backend — single-node RLinf alignment (opt-in; `ray_rlinf_alignment_implemented.md` has the full record)
- **Scheduler** (`dreamervla/scheduler/`): idempotent single-node `Cluster`/`ray.init`,
  `Worker`/`WorkerGroup` (broadcast, `execute_on`, send/recv), `placement.py`
  (Packed/Node/**Flexible** + accelerator-range syntax, ray-free testable), `channel.py`
  (actor FIFO + `get_batch` + `AsyncWork` no-wait), `node.py`, ray-free
  `WorkerManager`/`DeviceLockManager`, `dynamic_scheduler.py`, `hardware.py` (CUDA
  discover/validate, **no** auto batch/env), `collective/` (broadcast + tagged send/recv).
- **Workers** (`dreamervla/workers/`): per-env `EnvWorker`, batched `InferenceWorker`
  (encoder+WM+policy), `rollout_inference_worker` (OFT cold-start), `ReplayWorker`, `LearnerWorker`.
- **Learner loop** (`actor/learner_worker.py` + `runners/online_cotrain_ray_runner.py`): real
  `mode=dreamervla_cotrain` (wm/classifier/rl/cotrain phases reusing `online_dreamervla.py`);
  `synthetic_ppo` smoke; `OnlineCotrainRayRunner` (`online_cotrain_ray`) infer→step→learn
  overlap loop with deep ObjectRef double-buffering + `time/*` instrumentation.
- **FSDP/memory stack** (`hybrid_engines/fsdp/`): `FSDPModelManager` (FSDP1/**FSDP2**/cpu_offload/
  activation-checkpointing), pluggable `strategy/{base,fsdp,fsdp2,checkpoint}.py`, config-time
  autocast+GradScaler; `WORLD_SIZE=1` passthrough.
- **Weight-sync** (`hybrid_engines/weight_syncer/`): object-store default + `Collective` (NCCL) +
  `Bucket` + `Patch` (delta) + `Compressed` (fp16/bf16).
- **Decoupling**: `models/registry.py`, config-time precision validation, manual config groups
  `configs/{parallelism,precision,scheduler}/`, ops `start_ray.sh`/`check_ray.sh`.
- **Verified (2026-06-19, TDD)**: learner parity bit-exact across the actor boundary
  (`test_s5_learner_parity.py`); collective send/recv multi-channel; bucket/patch/compression
  sync; FSDP2 strategy; config fail-fast; real-component wiring + cold-start/online deep overlap +
  resource instrumentation; gated real e2e (`test_s5_ray_real_cotrain.py`, `test_s6_ray_real_oft_collect.py`).

## Cotrain / collector / rollout pipeline
- `OnlineCotrainPipelineRunner` (single-machine torchrun) = canonical **parity baseline** for the
  Ray backend; offline seed + warmup-split ckpt + launchers + light tests in repo.
- `CollectRolloutsRunner` (Hydra cold-start collector); parallel + in-card vectorized collection
  (`VecRolloutEnv` + batched OFT decoder); reward-HDF5 + obs_embedding sidecar +
  `preprocess_config.json` consumed zero-change by `BalancedTerminalDataset`.
- **Vectorized egl cotrain rollout** (`dreamer_image_from_record` + `build_cotrain_replay_transition`
  + `_vectorized_cotrain_rollout`, knobs `num_envs`/`render_backend`): merged `d25d0fc`, enable_grad
  fix `e23e7da`; validated e2e under osmesa (4-env ~6.4 env/s, warmup+RL+ckpt, clean exit). egl
  worker-level runtime verified on 2026-06-27: Ray 1 EnvWorker x4 children ran 160 env steps cleanly,
  disjoint no-Ray 4-env smoke passed, and overlap render/compute placement fails fast (`3f76ce4`).

## Performance optimizations merged (default-off or byte-identical unless noted)
| Item | What | Commit |
|---|---|---|
| W1/Q8 | `reduce_mean_dict` single all_reduce | `b58d782` |
| Q10 | `bin_centers` fancy-index decode | `6d340f7` |
| Q1 | EMA `_foreach_` fusion | `d9dced7` |
| Q2/Q6 | drop redundant CPU `.clone()` (device-conditional) + batched inference D2H | `ba18d0f` |
| Q3/Q4 | HDF5 slice-read actions | `29cf619` |
| Q5 | sparse-reward `scatter_` | `4eec644` |
| Q7 | offline DreamerV3 metric materialize gated behind `log_every` | `fac9302` |
| Q9 | `img2bpe` mapping on input device (per-device cache) | `8bae134` |
| Q11 | parallel bucket `ray.get` | `3094775` |
| readiness-gate + egl-wiring | skip per-step replay scan/all_reduce off `train_every`; drop forced `PYOPENGL_PLATFORM=egl` (SIGABRT cause) | `0e68754` |
| W2 | atomic temp→rename checkpoint save (caller-wiring still open → TODO.md) | `4aa4346` |
| H4/B | per-step grad-norm/cosine gated behind `optim.grad_diagnostics` (default OFF) | `889d3cb` |
| W6 | PPO actor backward micro-batched over B_eff (`lumos.update_micro_batch_starts`, default=original) | `016b900` |
| W3/W4 | manifest-first pretokenize index (guarded/dormant) + per-worker LRU frame cache | `6e09282` |
| H5 | switchable DINO-WM SDPA (`attn_impl`, default `manual`=byte-identical) | `d4d857a` |
| H9 | Chameleon `_update_causal_mask` cache (None-mask case) | `c69eb48` |
| prompt-tokenize cache | OFT rollout-hidden extraction caches invariant per-task prompt tokenization; tests cover once-per-task reuse and byte-equivalent cached text tensors | `6b8f366` |

## Other shipped features
- **Data-shard rotation + dual HF/torch checkpoints**: `demos_per_shard`; `training.checkpoint_format`
  (torch/hf/both) + generic `HFModuleWrapper` (`hf_module.py`) at 3 ckpt sites. NB: HF inner attr is
  `wrapped_module`.
- **ProgressReporter** (`utils/progress.py`) replacing all tqdm; **cotrain resume** via inherited
  base-checkpoint machinery.
- **Train-console output**: 3-layer console (files / runtime logs / `===` banners + metric box) +
  per-loop VLA-improvement line (`SuccessTracker`); unified `BaseRunner.console_*` API across runners.
- **MEM-RL-01** micro-batch LUMOS update (group-aligned, global-B_eff normalized = bit-for-bit
  full-batch gradient; knob `update_micro_batch_starts`, default off) — `816dd33`.
- **RUN-01 DDP code landed** `85788fc`: three default-off helper opt-ins + `online_dreamervla.main`
  rerouting + 7 unit tests; dist/checkpoint seams extracted into `_online_dreamervla_dist.py` /
  `_online_dreamervla_checkpoint.py` (GPU smoke is the open part → TODO.md).
- **PPO correctness + cleanup**: RLinf-alignment correctness audit done (A1 entropy-key + two approved
  numerics flips); dead-code removal (`models/chameleon_model/`, −5,625 LOC); suite 593 passed in the
  `dreamervla` env.
- **RLINF-01/02 partial**: RNG capture/restore + DreamerV3 consolidation + `utils/timers.py` helper
  landed (2026-06-22); wiring is open → TODO.md.
- **docs consolidation (2026-06-23)**: superpowers design specs deduped against `ray_*.md`; scattered
  `history/ plans/ specs/ superpowers/` records summarized into this file + `TODO.md`, referenced from
  AGENTS.md.
