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

- **ALG-03 (likely a real bug, not style):** the `outcome` PPO route uses
  `cfg.get("entropy_coef", 0.0)` and **drops the `actent` fallback** that
  `dense`/`dense_chunk` have. A config with `actent` set but not `entropy_coef`
  silently trains `outcome` with entropy 0. Needs a maintainer decision
  (intended vs bug) — not unifiable without changing behavior.
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
