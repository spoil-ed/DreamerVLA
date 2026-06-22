# TODO backlog (open items)

Open work only. Done items live in `../history/2026-06-21-backlog-execution-log.md`
(passes 1–3 + the 2026-06-21→22 GPU-box execution) and the other `docs/history/`
logs. This file is the only live open-items list.

## Core requirements (核心思想 — govern every item below)

1. **维持功能 / behaviour-preserving** — the #1 red line. Only merge code proven
   equivalent (AST/diff for identical, algebra/0.0-diff for math, seeded-batch for
   models); where implementations genuinely diverge, **flag it, do not silently
   unify**. Full unit suite stays green after every commit; anything that changes
   numerics is marked "changes numerics" and needs an explicit decision.
2. **统一实现 / one implementation per job** — the same functionality lives in ONE
   place; no competing or copy-pasted schemes. Make one canonical helper/interface,
   route all consumers through it.
3. **对齐 RLinf** — the upstream `RLinf` repo (workspace sibling) is the reference for
   RL correctness and overall code-tree alignment; diverge only deliberately.
4. **正确合理的接口** — algorithm/helper primitives have one correct, extensible
   interface (opt-in, default-off options); no lying/dead parameters.
5. **干净 + 简短可读** — minimal, surgical changes; no speculative/bloated code.

## Open items — all gated on the RynnVLA online path

The three remaining items all touch the **standalone RynnVLA `online_dreamervla` path**,
whose behaviour-changing parts the **unit suite cannot verify** (no multi-GPU DDP test) and
which **this box cannot run** (its WM/classifier ckpts — `outputs/worldmodel/
rynn_dino_wm_action_hidden/chunkaware_pinned/step_00017000.ckpt` + the outcome classifier — are
absent). Per core-req#1 they were not shipped blind. The clean seams already exist (extracted
2026-06-22, see history log): `_online_dreamervla_dist.py` and `_online_dreamervla_checkpoint.py`.

- [ ] **RUN-01 — route `online_dreamervla.main` DDP through the base helper.** Code-solvable:
  online_dreamervla wraps each *whole* module → maps to `helper.wrap_trainable_module`
  (not the per-child `wrap_world_model`). Needs **three default-off opt-ins** on
  `NopretokenizeSFTDistributedHelper` to preserve genuine divergences (so the mainline OFT
  callers stay byte-identical):
  1. `find_unused_parameters` — `True` for world_model/policy/critic, `False` for the classifier
     (helper currently hardcodes `False`).
  2. `broadcast_buffers` — online_dreamervla uses the DDP default **`True`**; the helper's
     `_wrap_module_with_ddp` hardcodes **`False`** (easy to miss — naive routing silently flips it).
  3. NCCL **timeout** — `DVLA_DDP_TIMEOUT_SEC` (helper's `initialize` has none).
  Keep the custom all-reduce error-wrapping (`_dist_all_reduce_flag/int`) — a deliberate divergence.
  **Verify with a RynnVLA multi-GPU save→resume smoke** before relying on it; suite-green ≠ verified.

- [ ] **X-01 (②, format-breaking remainder) — unify `online_dreamervla.save_checkpoint`.**
  Collapse its `{format_version, env_step, update_step (top-level), cfg, state_dicts}` into the
  canonical BaseRunner envelope `{format_version, cfg, state_dicts, pickles}`. This is a
  **multi-site format break**: top-level `env_step`/`update_step` is a consumer contract for
  `load_training_checkpoint`, `frozen_wm_actor_critic`, and three diagnostics
  (`measure_reward_and_drift` reads `ckpt["env_step"]` directly, `measure_wm_imagine_actor`,
  `measure_wm_imagine_fidelity`). Needs a dual-read loader + a **RynnVLA GPU save→resume** to prove
  old + new ckpts resume. (Resolved already: ① BaseRunner is the canonical writer and is
  GPU-verified; ③ the WM-only/classifier `{model,...}` payload is a genuinely-divergent inference
  artifact, flagged **not** unified per core-req#1. The shared dual-read `load_runner_payload` is
  in place.) Entangled with RUN-01 (BaseRunner-ifying `main()` is the clean route).

- [ ] **`online_dreamervla.py` `main()` split (P3).** The dist + checkpoint seams are already
  extracted (1861→1679). The remainder is `parse_args` (text-pinned by
  `test_online_env_episode_end`) and the 1264-line `main()` loop (unverifiable without the RynnVLA
  run). Do **after** RUN-01 + X-01② settle the DDP / save-load regions they rewrite.

## Won't-fix / intentional (record only)

**DIAG-06** (16 doc-only diagnostics) and **MOD-07** (`official` OFT action-model) — kept by
maintainer decision: not zero-import dead code (the diagnostics carry README rows + hygiene
test pins; `official` is called by `diagnostics/eval_openvla_oft_libero.py`) and they hold
paper/diagnostic value.
**Pixel-WM loss scaffolding** — assessed, genuinely diverges (CE vs MSE + extra backbone
hidden terms); not unified (see the pass-3 log).
ALG-02 (return assembly differs by rank/discount), UDA-06/04, MOD-05 (vendored OFT loader),
HF `register()` triplets (different classes per site), JSONL logging (`JsonLogger` drops
non-numeric fields — different job), RUN-09 (`build_optimizer` filters `requires_grad`),
`_decode_bpe` vs reconstructor, divergent diagnostics device-resolution groups, KL k1
signed estimator. See review/audit docs for rationale.
