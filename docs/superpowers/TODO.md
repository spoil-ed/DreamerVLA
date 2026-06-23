# TODO — open work (code master plan)

> The single forward plan: every open item, grouped by where it can run. Done/shipped work →
> [HISTORY.md](../HISTORY.md); architecture & rules → [AGENTS.md](../../AGENTS.md). When an item lands,
> move its line to `HISTORY.md` with the commit hash so this stays the live "what's next".

- Last updated: 2026-06-23
- Run unit tests in the **`dreamervla`** conda env (clean baseline ≈ 582–593 passed; base env gives
  ~13 spurious failures). The dev box **has 8×H100 but only intermittently** — GPU availability is the
  only real gate, not "missing ckpts / can't run".

---

## GPU-gated (needs a GPU box / LIBERO E2E)
- **RUN-01 smoke** — RynnVLA multi-GPU save→resume for the helper-routed online DDP. Code landed
  `85788fc`; open = run the multi-GPU save→resume + confirm the two `WORLD_SIZE=1`-only divergences
  (no PG built / no DDP-wrap for a single process) don't affect the real path.
- **COTRAIN-EGL** — verify the egl 4-env rollout (~4×) + clean exit on a free GPU. Core merged
  `d25d0fc` + readiness/egl-wiring `0e68754` validated under osmesa.
- **X-01② (format-breaking)** — unify `online_dreamervla.save_checkpoint` into the BaseRunner envelope
  `{format_version, cfg, state_dicts, pickles}`; top-level `env_step`/`update_step` is a consumer
  contract (`load_training_checkpoint`, `frozen_wm_actor_critic`, 3 diagnostics) → needs a dual-read
  loader + RynnVLA GPU save→resume. (① canonical-writer done+verified; ③ WM-only/classifier payload
  genuinely divergent, left alone.) Entangled with RUN-01.
- **RLINF-01 remainder** — prove multi-GPU RynnVLA save→resume is **bit-exact** (RNG); fold `rng` into
  the envelope when X-01 rewrites it. (RNG capture/restore already landed.)
- **RLINF-02 remainder** — route scattered `f"time/..."` points through the landed `Timers` helper +
  add default-off `Profiler`; all sites are in the GPU/Ray loops.
- **DECOUPLE-02** — route `OpenVLAOFTPolicy` / `RynnVLAEncoder` / `DreamerVLAOnlineTrainEnv` (hardcoded
  in `runners/online_utils`, `oft_collect_common`, `envs/train_env.py:694`, two `preprocess/*`) through
  `instantiate(cfg.<x>)`; GPU/LIBERO E2E to verify.
- **DECOUPLE-03** — inject `L1RegressionActionHead` (hardcoded in 3 actors + 1 encoder) via protocol +
  config `_target_`; deep model-internal, GPU-gated.
- **Perf W7** — dataloader `pin_memory`/`prefetch`/`non_blocking` for the dreamervla series; GPU smoke.
- **Perf H7** — WM autocast/GradScaler (replace static `.to(bf16)` no-scaler); GPU.
- **Perf H2** — replay contiguous per-field layout; DEFERRED (minimal version doubles replay host memory;
  full rewrite touches every consumer) → needs a real-run host-mem + throughput check.
- **Perf W5 single-forward** — replace OFT autoregressive `generate()` with one forward; PARKED
  (equivalence needs conditional-independence of action tokens, a real-model property) → real-OFT
  decode-equivalence smoke.
- **Perf W2 caller-wiring** — fold latest+top-k into one `save_checkpoint(extra_paths=...)`; PARKED
  (needs moving a `broadcast_object` before save, changes distributed-ckpt collective ordering). The
  atomic-write win is already active.
- **offline-warmup → online-cotrain real long-run** — pipeline + Ray-tiny both run; the real OFT/LIBERO
  convergence/metrics are unverified. Validate against the RLinf parallel-eval baseline (libero-goal
  traj1 ≈ 0.50 `success_once`); produce a reproducible command + run root + metrics summary. *(= the
  same task as Ray-todo item 1.)*
- **Perf benchmark** — no real throughput/memory benchmark yet; instrumentation landed, kernel on/off
  conclusions (FA2 / `torch.compile`; liger N/A on Chameleon) are gated on the long-run. "先量后调."
  *(= Ray-todo item 5.)*

## CPU-doable next (unit-testable here)
- **DECOUPLE-04** — small impls: `ReturnPercentileTracker` direct-instantiate; `BalancedTerminalDataset`
  runtime `cfg._target_` mutation (`frozen_wm_actor_critic.py:240`, `finetune_reward_head_sparse.py:160`)
  → into config; HF-save `target=` string → derive from config; move `soft_update` to a shared util.
- **Perf W8** — bf16 + `inference_mode` for frozen eval-only submodules (no-grad paths only).
- **Perf H3** — replay readiness incremental counts; marginal now the readiness-gate cut its frequency.
- **Perf H6** — WM KV-cache (follow-on to merged H5 SDPA); CPU equivalence-testable.
- **prompt-tokenize cache** — `rollout_hidden_extractor.py:230` cache the invariant per-task prompt
  tokenization (today the tokenizer runs `len(views)`× per step). Always-on, behavior-equivalent.
  *(in progress)*

## Structural refactors
- **MEM-RL-01 remainder + MEM-RL-02** — promote in-update imagination data (currently a local `slices`
  list in `dino_wmpo_outcome_step`) to its own explicit host buffer; then **WM-as-env** (RLinf/WoVR
  alignment): make the world model a gym env so imagination is a normal rollout into a separate replay +
  standard micro-batched PPO. Do together — MEM-RL-02 subsumes the remainder.
- **`online_dreamervla.main()` split (P3)** — dist+checkpoint seams already extracted (1861→1679);
  remainder = `parse_args` (text-pinned by `test_online_env_episode_end`) + the 1264-line `main()` loop.
  Do **after** RUN-01 + X-01② settle the DDP / save-load regions.

## Ray backend remaining (`ray_rlinf_alignment_implemented.md` is the shipped record)
- Items **1** (real LIBERO/OFT long-run) and **5** (benchmark-driven perf) above are the only open
  mainline Ray items.
- **Conditional P3 (trigger-only, default not done):** independent **reward/critic worker** (only if RL
  needs a separate reward/critic service — inline today); **hardware-registry + kernel** switches
  (`use_liger_kernel`/`attention_backend`, only for NPU/robot or kernel accel, default off);
  **Megatron/vLLM/SGLang** (only if model size/shape changes — single-card fixed-length forward →
  default non-goal). Channel async API = done.

## Non-goals — do NOT pursue
Multi-node horizontal scaling · VRAM auto-sizing / auto-batch / OOM-retry · collocated / disaggregated /
hybrid placement modes · Channel key-routing as a target.

## Won't-fix / intentional (record only, do not re-open)
DIAG-06 (doc-only diagnostics), MOD-07 (`official` OFT action-model), Pixel-WM loss scaffolding (genuinely
diverges), ALG-02, UDA-06/04, MOD-05 (vendored OFT loader), HF `register()` triplets, JSONL `JsonLogger`,
RUN-09, `_decode_bpe` vs reconstructor, divergent diagnostics device-resolution, KL k1 estimator.
`ChunkAwareDinoWMWorldModel(DinoWMWorldModel)` inheritance is already swappable (not a violation).
Optional **OFT Phase 5** generalization (object/spatial/10 suites) — same transformers fork + gripper fix
should carry over; not yet run per-suite.
