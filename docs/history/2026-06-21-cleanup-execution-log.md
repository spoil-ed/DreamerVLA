# Cleanup / Unification Execution Log

Date: 2026-06-21
Branch: `chore/cleanup-dead-code` (off `main` after the progress+resume feature landed).
Constraint honored throughout: **maintain functionality** — only behavior-preserving
changes; every divergence flagged, never silently unified. Full unit suite ran
green after every commit (`571 passed`, the 6 remaining failures pre-date this
work on `main`/`77fd789`).

## Done (one implementation, proven behavior-preserving)

| Commit | Item(s) | What |
|---|---|---|
| `refactor(models)` | MOD-01 | delete byte-identical dead `models/chameleon_model/` (−5,625 LOC), repoint 1 test |
| `chore(hygiene)` | CFG-01/02 | untrack `.planning/`, portable doc paths; greened the machine-path hygiene test |
| `chore(cleanup)` | UDA-07/PRE | dead `image_from_array`, orphan `del`, 3 live debug prints |
| `refactor(world_model)` | MOD-02 | RMSNorm/ChannelRMSNorm/`_module_*` → single source in `common.py` |
| `refactor(algorithms)` | ALG-01/04 | PPO clip term (4 copies, algebraically exact) → `grpo._ppo_clip_term`; lambda-return recurrence |
| `refactor(dataset)` | UDA-05 | HDF5 open/cache → `BaseDataset.cached_hdf5_file` |
| `refactor(actor)` | MOD-04/05 | actor checkpoint-extract → `actor/_load.py`; `module.`-strip → `hf_checkpoint.strip_module_prefix` |
| `refactor(embodiment)` | MOD-06 | dataset_statistics merge → `embodiment/_norm_stats.py` |
| `refactor(preprocess)` | PRE-01 | 3 byte-identical FlexAR methods → `_FlexARItemProcessorBase` (−112 LOC) |
| `refactor(viz)` | UDA-01 | `_safe_decode` + `_decode_gt_next` in `wm_image_viz` (visualize_batch 273→190) |
| `refactor(diagnostics)` | DIAG-02/03 | `slice_latent`/`reward_of` → `utils/latent.py`; `resolve_device` → `diagnostics/_common.py` |
| `refactor(world_model)` | MOD-03/12 | token-WM shared base `_dreamer_v3_token_common.py` (−392 LOC); `_reward_pred` deduped |

Each unification proven equivalent (AST/md5/`diff` for identical code, algebraic
proof + 0.0 numeric diff for math, seeded-batch behavior tests for models).
Where call sites differed, the shared helper was **parameterized to preserve each
exactly** (e.g. PPO reductions, actor `require_all_valid`, token-WM
`_dones_supports_is_terminal`).

## Flagged — NOT changed because it would alter behavior

- **ALG-03 (RESOLVED 2026-06-21):** all three PPO routes now read entropy via the
  shared `grpo._entropy_coef` (`actent` → `entropy_coef` → 0.0); the `outcome` route
  no longer drops the `actent` fallback. (Originally flagged: `outcome` used
  `cfg.get("entropy_coef", 0.0)`, dropping `actent`.)
- **DIAG-01 (WM-load):** `online_utils.load_world_model_state` is strictly *more*
  transformative than the inline diagnostic loaders (reward-head key remap +
  shape-mismatch skip, both live on real ckpts). Routing through it changes which
  weights load / crash-vs-skip. Needs a new opt-in `load_world_model_state_from_dict`
  variant — a design change, deferred.
- **ALG-02 (return/KL assembly):** structurally different (tensor ranks `[B,H]` vs
  `[B,H,K]` vs `[B,T]`; per-step vs no discount; differing KL placement).
- **UDA-06 / UDA-04:** outcome extraction genuinely diverges (reward-key
  precedence, per-window vs per-demo labels, return types); `VLASFTHDF5Dataset`
  can't subclass `BaseDataset` without adding a missing abstract method.
- **MOD-06 register triplets / MOD-05 vendored OFT loader / UDA-02 decoder:**
  register different classes per site / vendored fork code / non-equivalent VQ
  decode.
- **X-02 ad-hoc JSONL logging (≥12 sites):** NOT duplication of `JsonLogger` —
  `JsonLogger` drops non-numeric fields and resumes prior lines, while these sites
  log rich structured records (strings/dicts/lists) to per-purpose filenames with
  their own truncate semantics. Different job; left as-is.
- **Diagnostics device-resolution (no-guard / cpu-aware groups) and JSON tails:**
  divergent fallbacks / no two byte-identical.

## Needs a migration design (cannot be a pure refactor)

- **X-01 checkpoint unification** (3 payload schemes + 2 dir conventions across the
  dreamer/online runners): converging the schemes changes the **on-disk
  checkpoint format**, which would break resume/eval of existing checkpoints. A
  safe version needs a format-version + loader that reads both — a feature, not a
  refactor.
- **RUN-01** (dreamer runners reinventing distributed guards / dataloader): touches
  DDP/FSDP semantics; behavior-sensitive, warrants its own tested change.

## Out of scope here (structural smell, not duplication)

God-file decompositions (`embodied_eval_runner` 2514 LOC, `online_dreamervla`
`main` ~1265 LOC, `algorithms/dreamervla.py:imagine_actor_critic_step` ~834 LOC,
`pretokenize_dataset` 822 LOC) and MOD-07 (two intentional OFT action-model
implementations). Also noted by the token-WM pass: the pixel WMs share the same
loss scaffolding as the token WMs — a larger follow-on unification.

## Done — backlog execution pass 2 (2026-06-21, this branch)

Full unit suite green in the `dreamervla` env (**582 passed, 7 skipped**; baseline
was 571 passed + the 6 pre-existing failures, plus 5 new guard tests). All items
behaviour-preserving **except A2/A3**, an explicit maintainer-approved
`changes-numerics` opt-in. Corresponding backlog entries removed.

- **P0 green suite** — fixed the 6 pre-existing unit failures by repairing test
  mocks / script curation only (zero production change). `test_online_cotrain_pipeline`
  fake `_build_components` now sets `policy`/`critic`, pins `checkpoint_format=torch`,
  returns warmup losses, and pins the debug `total_env_steps` knob;
  `test_online_cotrain_ray_runner` gives the `__new__`-built runner a
  `NullMetricLogger`; `test_setup_scripts` + `scripts/README.md` register the new
  `collect_parallel.sh`. (Note: run unit tests in the `dreamervla` conda env —
  the base env's transformers version yields ~13 spurious failures.)
- **PRE-02 / PRE-03 / MOD-10 / RUN-14** — deleted verified-dead code: two orphan
  preprocess scripts; base `FlexARItemProcessor` + test-only
  `FlexARItemProcessorActionFast` (plus the now-unused `AutoProcessor` import,
  aliases, `__all__` entries, and the `test_preprocess_imports` asserts); the
  always-raising `OFTActionHiddenEncoder` placeholder (+ `encoder/__init__`
  re-export); the unwired `online_dreamervla_multiproc.py` fork (+ AGENTS.md /
  `scripts/README.md` mentions).
- **CFG-04** — `configs/task/_base_libero.yaml` + four thin suite files
  (468 → 206 lines). Verified: all 13 task configs (incl. the `OpenVLA_Onetraj_*`,
  `ColdStart_*`, `RynnVLA_*` inheritance chain) resolve byte-identically vs the
  pre-change hydra compose.
- **ppo_gamma/lam (audit verify)** — every RL config sets `ppo_gamma: 1.0` +
  `lam: 0.95` explicitly (deliberate: `lam` matches RLinf, `gamma=1.0` is intended
  for the episodic sparse-outcome return). No change needed.
- **A2/A3 numerical-stability guards (changes-numerics, approved)** — enabled the
  RLinf-aligned dual-clip `clip_ratio_c: 3.0` and the summed-trajectory log-ratio
  clamp `clip_log_ratio: 10.0` in all 6 RL `algorithm:` blocks (the `dense` /
  `dense_chunk` / `outcome` routes already read both keys; threaded them through the
  `relabel` route too). Both are no-ops at the default (None); the dual-clip
  primitive is algebraically identical to RLinf's `min(loss, c·|adv|)`. Locked by
  new `test_ppo_clip_guards.py` (5 tests incl. default-off equivalence). The
  `clip_log_ratio` value is a conservative anti-overflow bound (RLinf has no
  per-trajectory analogue — it sums per-token); tune to the observed distribution.
- **Rename `_real_relabel_ppo_loss` → `_real_relabel_anchor_loss`** — it is a
  frozen-old-logprob, constant-advantage BC anchor, not on-policy PPO (4 internal
  refs; the registry `ppo` route alias is unchanged, so configs are unaffected).
- **CFG-06 (already done)** — the `OpenVLA_Onetraj_*` task configs and the rynnvla
  `*_input_token_chunk` classifier already compose thin `defaults:`; nothing to do.
