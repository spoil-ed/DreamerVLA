# Performance Audit — Execution Roadmap

> Companion to `docs/plans/performance_optimization_audit.md` (the **findings**). This doc is the
> **execution layer**: priority framing, phased order, test gates, and per-item status tracking.
>
> **Standing rule (user):** every work-item below gets its OWN complete, detailed plan in
> `docs/plans/` *before* any code is written. This roadmap is the index of those plans, not a
> substitute for them. All tests/ruff/GPU smoke run in the **dreamervla** conda env.

---

## 0. Framing — the highest-value theme: "已实现,只是没串进主线"

The audit's own one-line summary: *"很多优化就是把已有的好写法搬到还没用上的路径"*. These **wire-in**
items are the priority — the code/pattern ALREADY EXISTS in the repo (so it's low-risk and often
already tested); the work is connecting it to the path that hasn't adopted it.

| ID | Existing implementation / pattern (already in repo) | Target path that hasn't adopted it | Audit ref | Risk |
|----|-----------------------------------------------------|------------------------------------|-----------|:----:|
| **W0** | `VecRolloutEnv` + batched decoder (collection path) | cotrain online rollout (single-env osmesa) | H1 | med — **= Option 1, IN FLIGHT** |
| **W1** | batched all_reduce at `dreamerv3_pixel_runner.py:262` | `distributed.py:185-213 reduce_mean_dict` (per-key all_reduce) | Q8 | low |
| **W2** | atomic temp-then-rename at `_dreamer_runner_common.py:86` | `base_runner.py:800-821,1054` checkpoint save | H8 | low-med |
| **W3** | manifest-first indexing at `one_trajectory_pretokenize_dataset.py:88` | `pretokenize_dataset.py:300-343` (re-pickles every file in `__init__`) | §3.6/F | low |
| **W4** | per-worker `cached_hdf5_file` handle cache (pixel_sequence / vla_sft) | datasets lacking it (pretokenize per-frame, wm_replay_classifier) | §3.6/F | low |
| **W5** | "official" OFT/openvla variant: `bin_centers[idx]`, slice-256-bin, single forward | DreamerVLA-adapter variant `openvla_action_model.py:664-728,699,722` | Q10/§3.5 | low-med |
| **W6** | micro-batch memory pattern in `outcome.py` (`update_micro_batch_starts`, `_slice_latent`, per-chunk backward) | `dense.py:580-706` / `dense_chunk.py:290-332` (whole-trajectory single backward → OOM surface) | §5 third batch | **med-high — needs equivalence test** |
| **W7** | `pin_memory:true` + `non_blocking=True` in `dreamerv3_pixel/token` configs | dreamervla-series config + `dreamervla_runner.py` H2D | E | low — needs smoke |
| **W8** | `item_processor.py` `@no_grad` / `inference_mode` patterns | other preprocess/encoder paths still fp32+`no_grad` | §3.6/D | low-med |

These W-items, plus the cotrain hot-path micro-fixes (readiness-gate, prompt-cache), are the spine of
this program.

---

## 1. Phased execution order (risk axis from audit §5; ⭐ = wire-in from §0)

> Each phase = a batch of work-items. We do them in order; within a phase, ⭐ wire-in items first.
> A phase does not start until the prior phase's items are merged or explicitly deferred.

### Phase 0 — in flight
- **W0 / Option 1** — cotrain vectorized egl rollout. Plan: `2026-06-23-cotrain-vec-egl-rollout.md`.
  Fork implementing in a worktree; parent does GPU smoke + merge.

### Phase 1 — numerically-equivalent quick wins (audit 第一批, low risk)
⭐ **W1** (Q8 reduce_mean_dict), ⭐ **W5** (Q10 bin decode + §3.5 single-forward), then the rest:
Q1 (`ema.py` `_foreach_`), Q2 (drop redundant `.clone()` ×5), Q3/Q4 (HDF5 slice-read instead of
whole-`actions`), Q5 (`outcome.py` sparse-reward `scatter_`), Q6 (`inference_worker` batched D2H),
Q7 (gate metric `.cpu()` materialization behind `log_every`), Q9 (`img2bpe` as GPU buffer),
Q11 (parallel `ray.get` in `bucket.py`), **cotrain readiness-gate** (`online_cotrain_runner.py:585`
→ compute `get_replay_task_stats_global` only at `train_every` boundaries; behavior-equivalent),
**prompt-tokenize cache** (`rollout_hidden_extractor.py:230`).
- Gate: numerically/behaviorally equivalent → unit test where one exists; full suite + ruff green.

### Phase 2 — needs GPU smoke (audit 第二批)
⭐ **W2** (H8 atomic checkpoint + background save), ⭐ **W7** (E: dataloader `pin_memory`/`prefetch`
+ `non_blocking`), B (diagnostics/grad-decompose gated behind `debug`/`log_every`), H3 (replay
readiness incremental + per-N-step, beyond the cotrain gate).
- Gate: a `debug=true` GPU smoke (80GB is near-OOM at full — keep headroom, see [[oft-online-cotrain-default-debug]]).

### Phase 3 — equivalence-test + refactor (audit 第三批, higher risk)
⭐ **W6** (dense.py/dense_chunk.py ← outcome.py micro-batch), ⭐ **W3/W4** (dataset manifest-first +
handle cache — low-risk but touches IO contracts), ⭐ **W8** (bf16+`inference_mode` for frozen
eval-only submodules), H2 (`online_replay` → per-field contiguous arrays), H5/H6 (WM KV-cache /
`scaled_dot_product_attention`), H7 (autocast/GradScaler), H9 (Chameleon mask O(L²)→cached).
- Gate: an equivalence unit test modeled on `tests/unit_tests/test_wmpo_microbatch_equivalence.py`
  behind a switchable path; + GPU smoke. Honor the §4 constraints (1:1-upstream numerics, vendored 🔒,
  fire-and-forget Ray backpressure).

---

## 2. Per-item status tracking

> Updated as we go. "Plan" = the item's own detailed `docs/plans/` doc (written before implementation).

| Item | Phase | Plan doc | Status | Commit |
|------|:-----:|----------|--------|--------|
| W0 / Option 1 | 0 | `2026-06-23-cotrain-vec-egl-rollout.md` | reviewed SAFE, **MERGED** d25d0fc; **validated e2e** (osmesa 4-env ~6.4 env/s, warmup+RL bursts+ckpt, clean exit); egl runtime pending free GPU | d25d0fc |
| Option-1 enable_grad fix | 0 | `2026-06-23-cotrain-vec-egl-rollout.md` | **MERGED** (rollout @no_grad wrapped the burst → fixed) | e23e7da |
| egl-wiring fix (drop forced PYOPENGL_PLATFORM=egl) | 0 | `2026-06-23-cotrain-readiness-gate-and-egl-wiring.md` | **MERGED**; egl runtime verify pending free GPU | 0e68754 |
| W1 reduce_mean_dict | 1 | `2026-06-23-perf-w1-reduce-mean-dict.md` | **MERGED** | b58d782 |
| W5 bin decode (Q10 done) / single-forward (todo) | 1 | `2026-06-23-perf-q10-bin-centers-vectorize.md` | Q10 **MERGED**; single-forward not started | 6d340f7 |
| Q1 ema _foreach_ | 1 | `2026-06-23-perf-q1-ema-foreach.md` | **MERGED** | d9dced7 |
| Q2 drop .clone() | 1 | `2026-06-23-perf-q2q6-clone-d2h.md` | **MERGED** (device-conditional `_independent_cpu`) | ba18d0f |
| Q3/Q4 HDF5 slice-read | 1 | `2026-06-23-perf-q3q4-hdf5-slice-read.md` | **MERGED** | 29cf619 |
| Q5 sparse-reward scatter | 1 | `2026-06-23-perf-q5-sparse-reward-scatter.md` | **MERGED** | 4eec644 |
| Q6 batched D2H | 1 | `2026-06-23-perf-q2q6-clone-d2h.md` | **MERGED** | ba18d0f |
| Q7 metric materialize gate | 1 | `2026-06-23-perf-q7-metric-materialize-gate.md` | **MERGED** fac9302 (offline dreamerv3 pixel+token runners: per-step `grad_norm`/metric `.cpu()` D2H moved inside the existing `log_every` gate; logged values byte-identical, `_loss` backward target left on-device) | fac9302 |
| Q9 img2bpe GPU buffer | 1 | `2026-06-23-perf-q9-img2bpe-device-buffer.md` | **MERGED** 8bae134 — `convert_img2bpe` now gathers on the input's own device (per-device mapping cache), killing the per-call `.to("cpu")` D2H + H2D round-trip; tokens byte-identical. NB: `ChameleonImageVocabularyMapping` is a plain class (not `nn.Module`), so the planned `register_buffer(persistent=False)` was infeasible → device-cache used instead; no `state_dict` keys added | 8bae134 |
| Q11 parallel ray.get | 1 | `2026-06-23-perf-q11-bucket-parallel-rayget.md` | **MERGED** | 3094775 |
| cotrain readiness-gate | 1 | `2026-06-23-cotrain-readiness-gate-and-egl-wiring.md` | **MERGED** | 0e68754 |
| prompt-tokenize cache | 1 | `2026-06-23-perf-prompt-tokenize-cache.md` | agent in progress | — |
| W2 atomic checkpoint | 2 | `2026-06-23-perf-w2-atomic-checkpoint.md` | **MERGED** 4aa4346 (atomic temp→rename active for every save; single-serialize `extra_paths` capability added+tested but call-sites in `openvla_oft_runner.py`/`pretokenize_vla_runner.py` not yet rewired — deliberate strict-scope deviation) | 4aa4346 |
| W7 dataloader/non_blocking | 2 | _tbd_ | not started | — |
| B / H4 grad-diagnostics gating | 2 | `2026-06-23-perf-h4-gate-grad-diagnostics.md` | **MERGED** 889d3cb (per-step actor grad-norm/cosine extra-backwards gated behind `optim.grad_diagnostics`, default OFF; OFF math bit-identical, atol=0). Broader B (other per-step diagnostics/sync, if any) still TBD | 889d3cb |
| H3 replay readiness incremental | 2 | _tbd_ | not started | — |
| W6 dense ← outcome micro-batch | 3 | `2026-06-23-perf-w6-dense-microbatch.md` | agent in progress | — |
| W3 manifest-first | 3 | `2026-06-23-perf-w3w4-pretokenize-io.md` | **MERGED** 6e09282 — but GUARDED/dormant: shipped manifests store the *next*-frame image, so the manifest-first path falls back to the (unchanged) pickle scan; byte-identity preserved, no perf win until the preprocess writer also emits the current-obs image (out of scope) | 6e09282 |
| W4 hdf5 handle cache | 3 | `2026-06-23-perf-w3w4-pretokenize-io.md` | **MERGED** 6e09282 (per-worker bounded-LRU frame cache `_load_frame_payload`, cap 64; overlapping stride-1 windows unpickle a shared frame once; byte-identical) | 6e09282 |
| W8 bf16 frozen eval | 3 | _tbd_ | not started | — |
| H2 replay contiguous layout | 3 | _tbd_ | not started | — |
| H5/H6 WM KV-cache / SDPA | 3 | _tbd_ | not started | — |
| H7 autocast/GradScaler | 3 | _tbd_ | not started | — |
| H9 chameleon mask | 3 | _tbd_ | not started | — |

---

## 3. Execution protocol (per work-item)

1. Write the item's detailed plan in `docs/plans/` (problem → existing pattern to reuse → exact
   files:lines → TDD steps → test/smoke gate), referencing the audit § for the finding. Get sign-off.
2. Implement TDD. For any numerics/sampling/precision/vendored touch, add a switchable path +
   equivalence test (model: `test_wmpo_microbatch_equivalence.py`). For config/memory/precision,
   add a `debug=true` GPU smoke.
3. Full unit suite + `ruff` green in dreamervla. Separate commit per item, conventional + `--signoff`,
   no `===`/`/` in subject. Update §2 status row.
4. Cluster trivially-related equivalent items (e.g. several Q-items in one file) into one plan/commit
   when it reduces churn — but never bundle a numerics-touching item with an unrelated one.

## 4. Notes
- §0 W-items are the priority because the implementation already exists — verify the existing pattern
  still matches the target's contract before wiring (the audit's row is a lead, not a guarantee).
- The cotrain-throughput critical path (the user's active goal) is: W0 (egl) → readiness-gate +
  prompt-cache → H2 (replay sample) → B (per-step diagnostics/sync). Other phases serve offline
  training / Ray / dataset paths and can be scheduled by which path is being pushed.
- Do NOT launch a multi-agent Workflow for this without explicit user opt-in; per-item forks/Explores
  via the Agent tool are the default execution vehicle.
