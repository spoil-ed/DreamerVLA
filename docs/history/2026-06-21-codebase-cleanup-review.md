# DreamerVLA Codebase Cleanup Review

Date: 2026-06-21
Scope: whole repository (`dreamervla/` ~80k LOC across 230 Python files + 107 configs).
Method: 6 parallel read-only subsystem audits (runners, models, preprocess,
utils/dataset/algorithms, diagnostics/envs, configs/hygiene). Every "dead" /
"duplicate" claim was grep-verified for importers before listing; corrected
false positives are recorded at the end. **Nothing was edited — this is a review.**

How to read: items are numbered per subsystem (`RUN-*`, `MOD-*`, `PRE-*`,
`UDA-*`, `DIAG-*`, `CFG-*`) for line-by-line tracking. Each item is
`[dimension][severity] location — issue. Fix: …`. Dimensions: **DUP**
(duplication), **INCONSISTENT** (competing schemes for one job), **DEAD**
(unused), **SMELL** (oversized/tangled), **HARDCODE** (magic literals/paths),
**DEBUG** (leftover prints/scaffolding).

---

## Executive summary

**One meta-theme runs through every subsystem: the shared infrastructure already
exists, but it is only half-adopted, so a second (or third) parallel scheme for
the same job lives beside it.** This is exactly the "one approach in front,
another behind" non-uniformity to remove. The five clearest instances:

1. **A whole duplicate vendored model tree** — `models/chameleon_model/` is a
   byte-identical copy of `models/embodiment/chameleon_model/` with **zero
   runtime importers** (5,625 LOC). [MOD-01]
2. **Checkpoint I/O has three payload schemes and two directory conventions**;
   the dreamer/online runners reinvent what `BaseRunner.save_checkpoint` +
   `get_checkpoint_dir()` already provide. [RUN-02, RUN-03]
3. **The dataset dir has two competing families** — only 3 of the dataset
   classes use `BaseDataset`; the rest re-implement HDF5 caching, demo-key
   listing, and outcome extraction. [UDA-04..06]
4. **`item_processor.py` is ~80% copy-paste across four classes** (~930 LOC that
   could be ~350). [PRE-01]
5. **The magic token id `10004` / `model_dim 4106` are baked in at 45 / 8 sites**
   though a validated config surface already exists — a half-finished config
   migration. [MOD-09, CFG-07]

### Highest-ROI items (do these first)

| # | Item | Impact | Effort | Risk |
|---|------|--------|--------|------|
| MOD-01 | Delete dead duplicate `models/chameleon_model/` (repoint 1 test) | −5,625 LOC | low | low |
| CFG-02 | Untrack root scratch files + `.planning/`; extend `.gitignore` | removes only machine-path leaks | low | low |
| PRE-01 | Collapse 4-way `item_processor.py` class dup via a mixin | ~−500 LOC | med | med |
| RUN-01 | Route dreamer/online runners through `BaseRunner`+`distributed` | kills 3 reinvented subsystems | high | med |
| UDA-04 | Unify dataset classes under `BaseDataset` | kills 3× HDF5 + outcome dup | med | med |
| CFG-04 | `configs/task/_base_libero.yaml` for the 4 copy-pasted suites | ~−350 cfg lines | low | low |
| MOD-02 | Adopt `world_model/common.py` (stop re-declaring 6 helpers) | de-drift | low | low |
| DIAG-01 | Route WM-loading diagnostics through `online_utils.load_world_model_state` | ~−150 LOC | low | low |
| ALG-01 | Shared `_ppo_clip_loss` for the 3 divergent PPO forms | fixes silent skew | low | med |
| MOD-09/CFG-07 | Finish `10004` / `4106` config migration | de-drift | med | low |

### Safe to execute immediately (zero design decision)

MOD-01 (delete verified-dead duplicate + repoint `test_repository_hygiene.py:380`),
CFG-01/CFG-02 (untrack scratch/planning files, extend `.gitignore`), and the
small dead-symbol removals (UDA-07 `image_from_array`, MOD-10
`OFTActionHiddenEncoder` if confirmed, PRE-13 unused `json` import,
RUN-DEBUG arrow-glyph). Everything else needs a (small) refactor or a
maintainer keep/delete decision.

---

## Cross-cutting (unification) issues

- **X-01 [INCONSISTENT][high]** Three checkpoint payload schemes coexist:
  `BaseRunner.save_checkpoint` (`{cfg,state_dicts,pickles}`, FSDP-aware),
  the dreamer runners' flat `{model,optimizer,rng,cfg}`
  (`dreamerv3_pixel_runner.py:140-164`, dup in token), and the top-level
  `online_dreamervla.save_checkpoint()` function (`:476-521`). Plus two dir
  conventions: canonical `checkpoints/` vs hardcoded `ckpt/`
  (`dreamerv3_pixel_runner.py:60`, `dreamerv3_token_runner.py:41`). Converge on
  the base API + `get_checkpoint_dir()`.
- **X-02 [INCONSISTENT][high]** Ad-hoc JSONL logging (`open(*_logs.json.txt)` +
  `json.dumps(row)+"\n"`) reimplemented in ≥9 runners alongside the canonical
  `MetricLogger`/`JsonLogger` + `BaseRunner.log_metrics`. Standardize on
  `JsonLogger`.
- **X-03 [HARDCODE][high]** Magic token id `10004` at ~45 sites and
  `model_dim 4106` at 8 config sites, despite existing config surfaces
  (`env.target_token_id`, validated `token_dim+action_emb_dim*repeat`). Finish
  the migration; no bare literals.
- **X-04 [INCONSISTENT][med]** Checkpoint-/component-loading boilerplate
  (prefix-strip, `state_dict`/`model`/`state_dicts.*` discovery, dtype cast) is
  copy-pasted across actors, encoders, and ≥12 diagnostics scripts. One shared
  `load_component_state_dict` + `strip_module_prefix` helper.

---

## Runners (`dreamervla/runners/`)

- **RUN-01 [INCONSISTENT][high]** `dreamerv3_pixel_runner.py:44-257` (+ subclasses
  `latent_wm_runner`, `backbone_dreamerv3_wm_runner`) extends `BaseRunner` but
  rolls its own `is_main_process`/`_print`/`_barrier`/DDP init/device/`_make_loader`/
  checkpoint I/O instead of `self.distributed` + base helpers. Only 4 runners use
  the canonical `distributed` helper. Fix: initialize `self.distributed` and drop
  the hand-rolled guards/loader.
- **RUN-02 [INCONSISTENT][high]** Checkpoint dir split (see X-01): route the
  dreamer + `online_dreamervla` writers through `get_checkpoint_dir()`.
- **RUN-03 [INCONSISTENT][high]** Checkpoint payload schemes (see X-01): converge
  onto `BaseRunner.save_checkpoint` or document why a flat format is required.
- **RUN-04 [INCONSISTENT][high]** `online_dreamervla.py` / `_multiproc.py` are
  249-line-argparse `main()` scripts bypassing Hydra/`BaseRunner`, while
  `online_cotrain_runner.py` is a Hydra runner — two config systems for one
  family. Migrate to the Runner/Hydra pattern.
- **RUN-05 [DUP][high]** `_save_ckpt`/`_resolve_resume_path`/`_maybe_resume`
  near-verbatim between `dreamerv3_pixel_runner.py:140-257` and
  `dreamerv3_token_runner.py:68-167` (token even misses pixel's
  `resume_reset_step`). Lift to a base/mixin.
- **RUN-06 [DUP][high]** Viz-strip / VQGAN viz reimplemented across
  `dreamerv3_token_runner.py:169-258`, `dreamerv3_pixel_runner.py:276-304`,
  `dreamervla_runner.py:632-723` (+ `_maybe_save_viz` overridden in 5 files).
  Extract a shared viz utility.
- **RUN-07 [DUP][high]** Three rollout-collection schemes share no code
  (`cold_start_ray_collect`, `collect_parallel_rollouts`,
  `collect_online_rollouts_for_classifier`); `oft_collect_common.py` covers
  policy/action loading but not env factory / rank sharding / HDF5 dump. Grow it
  into the real shared base; `collect_online_rollouts_for_classifier` and the
  multiproc fork can then likely retire.
- **RUN-08 [DUP][med]** `_to_device` identical in `dreamerv3_pixel_runner.py:20-26`
  and `dreamerv3_token_runner.py:19-25`; `_json_safe` duplicated with *divergent*
  bodies in `online_dreamervla.py:124-140` vs `_multiproc.py:125-135` (multiproc
  drops tensor/ndarray handling — latent bug). One impl in `online_utils.py`.
- **RUN-09 [DUP][med]** AdamW construction with identical defaults at
  `dreamerv3_pixel_runner.py:474`, `dreamerv3_token_runner.py:393`,
  `latent_classifier_runner.py:209`, `online_dreamervla.py:825` — a
  `build_optimizer(cfg)` helper already exists for the cotrain path; reuse it.
- **RUN-10 [DUP][med]** `backbone_dreamerv3_wm_runner.py:55-64`
  `_build_frozen_encoder_cfg(self)` (no `cfg`) **shadows**
  `BaseRunner._build_frozen_encoder_cfg(self, cfg)` (`base_runner.py:311`) and
  silently does something different (doesn't freeze). Rename or reuse the base.
- **RUN-11 [SMELL][high]** `embodied_eval_runner.py` is a 2514-line, ~50-method
  god-class mixing VLA/Dreamer/online-RSSM/TDMPC eval + trace + relabel export;
  worst methods `_evaluate_libero_online_rssm` (344 lines) and `_run_dreamer_eval`
  (262). Its docstring claims "exactly one" eval path but it overrides
  `evaluate_libero` to add a second. `"feat" in locals()` checks (`:1754-1807`)
  are refactor-pressure smells. Split eval modes into modules.
- **RUN-12 [SMELL][high]** `online_dreamervla.py:592-1856` `main()` ~1265 lines;
  `dreamervla_runner.py run()` ~789 lines (`:1322-2111`). Extract phase
  functions.
- **RUN-13 [SMELL][med]** `cold_start_ray_collect_runner.py` `_run_loop`
  (250-377) + `_run_loop_overlap` (379-579) are two overlapping ~328-line loops.
  Extract one `RayCollectionDriver`.
- **RUN-14 [DEAD/SMELL][med]** `online_dreamervla_multiproc.py` (581 LOC) is an
  unmaintained near-fork (no test, no registry, only a `python -m` mention in
  `scripts/README.md`/`AGENTS.md`). Decide: wire in, archive, or delete.
- **RUN-15 [SMELL][med]** argparse `__main__` tools live in `runners/` among
  Runner classes (`frozen_wm_actor_critic`, `rlinf_libero_rollout`,
  `collect_online_rollouts_for_classifier`, `online_dreamervla*`). Move to
  `tools/`/`scripts/` or document the runner-vs-script split.
- **RUN-16 [DEAD][low]** `collect_parallel_rollouts.py:58-61` module-level alias
  shadows of `oft_collect_common` imports — drop, use imports directly.
- **RUN-17 [HARDCODE][med]** `target_token_id = 10004` (5+ sites in
  `embodied_eval_runner.py`, argparse default `online_dreamervla.py:277`); action
  slice `[:7]` ~10× in `embodied_eval_runner.py`. Config-source once. (part of X-03)
- **RUN-18 [DEBUG][low]** `embodied_eval_runner.py:370-371` truncate-then-append
  scaffolding (`open(...,"w"): pass`); arrow-glyph inconsistency `→` (`:184`) vs
  `->` (`:494`).

## Models (`dreamervla/models/`)

- **MOD-01 [DEAD][high]** `dreamervla/models/chameleon_model/` (13 files, 5,625
  LOC) is a byte-identical duplicate of `models/embodiment/chameleon_model/`
  (`diff -rq` = 0 diffs) with **zero importers**. Pinned only by a stale test
  `test_repository_hygiene.py:380`. Fix: delete the tree, repoint that test at the
  `embodiment/` `__init__.py`.
- **MOD-02 [DUP][high]** `world_model/common.py` defines `RMSNorm/ChannelRMSNorm/
  act/_module_ref_tensor/_module_dtype/_module_device` but only `MLPHead` is
  imported from it; the other six are re-declared byte-identically in
  `base_world_model.py:19-45` and `dreamerv3_torch.py:24-86`. Import from
  `common.py` everywhere; delete the duplicates.
- **MOD-03 [DUP][high]** `dreamer_v3_token_world_model.py` vs
  `dreamer_v3_token_from_pixel_world_model.py` — both 214 lines, ~70% identical
  (only ~62 diff lines: class name + `is_terminal`/`dones` fallback + one kwarg).
  Extract a shared base/mixin.
- **MOD-04 [DUP][high]** Actor checkpoint-load blocks (~60 lines: prefix-strip +
  `state_dict`/`model`/`state_dicts.encoder` discovery + dtype cast) near-identical
  in `actor/latent_to_action_hidden_actor.py:198-259`,
  `actor/rynnvla_action_hidden_actor.py:115-206`,
  `actor/vla_action_head_actor.py:99-150`, `openvla_discrete_token_actor.py:151-191`.
  One shared `load_component_state_dict`. (part of X-04)
- **MOD-05 [DUP/INCONSISTENT][med]** `module.`-prefix stripping done 3 ways:
  `openvla_discrete_token_actor.py:17` (`_strip_prefixes`),
  `encoder/openvla_oft_policy.py:26` (`key[7:]`),
  `embodiment/openvla_oft/openvla_utils.py:110` (inline). One helper.
- **MOD-06 [DUP][med]** HF-loader boilerplate copy-pasted across embodiment
  `__init__.py` loaders: the `Auto*.register("openvla", ...)` triplet
  (`embodiment/openvla/__init__.py:36-38`, `openvla_oft/dreamervla/__init__.py:40-42`,
  `openvla_oft/official/__init__.py:51-53`) and the `dataset_statistics.json`→
  `norm_stats` merge (4+ copies). Shared `register_openvla_auto_classes()` +
  `merge_norm_stats()`.
- **MOD-07 [INCONSISTENT][med]** Two divergent OFT action-model impls
  (`embodiment/openvla_oft/{official,dreamervla}/openvla_oft_action_model.py`,
  ~1,020-line diff) selected by `implement_version`. Both reachable (`official`
  used by `diagnostics/eval_openvla_oft_libero.py`). Maintainer decision: retire
  `official`? (The shared `policy.py` ABC is good — leave it.)
- **MOD-08 [SMELL][med]** `dino_wm_chunk.py:380-552` `chunk_loss()` ~170-line
  6-deep-nested monolith; `dreamer_v3_pixel_backbone_world_model.py:88-216`
  ~128-line `__init__` with an 11-branch `hidden_decoder_kind` dispatch → factory.
- **MOD-09 [HARDCODE][high]** Magic `10004` at 45 sites total (models + runners);
  config surface (`env.target_token_id`, dataclass default `train_env.py:82`)
  exists but most sites bake the literal. Finish the migration. (X-03)
- **MOD-10 [DEAD][med]** `encoder/oft_action_hidden_encoder.py` — self-described
  "Placeholder", `encode()` always raises `NotImplementedError`, zero consumers
  (kept alive only by the `encoder/__init__.py:2` re-export). Drop or mark
  scaffolding.
- **MOD-11 [SMELL][low]** `encoder/openvla_oft_policy.py` is a 338-line *trainable
  policy* under `encoder/` (semantic mismatch) — relocate to `embodiment/` or a
  `policies/` package.
- **MOD-12 [DUP][low]** `_reward_pred` identical in `base_world_model.py:42-45`
  and `reward_heads.py:180-183`. Import from one place.

## Preprocess (`dreamervla/preprocess/`)

- **PRE-01 [DUP][high]** `item_processor.py:27-919` — four `FlexARItemProcessor*`
  classes share ~80% verbatim bodies (`__init__` VAE setup ×4, `process_image`,
  `get_n_grids_token`, `token2id`, `process_item`, `decode_image`). ~930→~350 LOC
  via a `_ChameleonImageMixin` base.
- **PRE-02 [DEAD][high]** `pretoken_world_model.py` (141 LOC) +
  `world_model_bi_views_conv_generation.py` (243 LOC) — 0 external references;
  superseded by `pretoken_state_action_model.py` /
  `action_state_model_conv_generation.py` (`with_world_model` flag). Confirm +
  delete (~384 LOC).
- **PRE-03 [DEAD][high]** `item_processor.py` base `FlexARItemProcessor`
  (`:27-193`, 0 real uses) and `FlexARItemProcessorActionFast` (`:712-914`,
  test-only). Verify + drop (~270 LOC).
- **PRE-04 [DUP][med]** `pre_tokenize_action_local.py` vs `_state_local.py` ~75-80%
  identical, and the *simpler* one imports `build_wm_action_mask`/`ensure_next_obs`
  **from** the complex one (backwards coupling). Lift a shared
  `_pretoken_worker.py` parameterized on `with_state`.
- **PRE-05 [HARDCODE][high]** `item_processor.py:727,759,766-767`
  (`FlexARItemProcessorActionFast`) hardcodes
  `"../checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768/..."`,
  `"./checkpoints/chameleon/tokenizer/..."`, `device="cuda"` instead of the
  `paths.py` constants its 3 siblings use. Route through `paths.py`.
- **PRE-06 [DUP][med]** `regenerate_libero_dataset_filter_no_op.py` (340) vs
  `regenerate_libero_failure_demos.py` (326) — env reset+settle, replay loop, HDF5
  obs-group write ~90-95% identical. Extract `reset_and_settle`/`replay_demo`/
  `write_hdf5_episode` into `libero_utils.py`.
- **PRE-07 [DUP][med]** 7 shared HDF5/path helpers (`_project_path`,
  `_demo_sort_key` byte-identical ×3; `_list_demo_keys`, `_state_from_obs_group`,
  `_history_indices`, `_image_from_hdf5`, `_task_prompt_from_path` ×2) across the
  3 `preprocess_*_hidden.py` sidecar scripts → `artifact_utils.py`. (NB: the two
  ~1000-line files are *not* mostly dup of each other — bulk is real model
  inference.)
- **PRE-08 [DUP][med]** `_format_percent` byte-identical and `_all_exist`
  near-clone in `validate_convs.py:123-128` / `validate_pretokenized.py:184-189` →
  shared `validate_common.py`.
- **PRE-09 [INCONSISTENT][med]** CLI parsing split: 12 files use the standard
  `script_namespace(...)`, 10 still raw `argparse`. Migrate the live ones.
- **PRE-10 [DEBUG][med]** `item_processor.py:686,691,695`
  (`FlexARItemProcessorActionState.decode_image`) has **live** `print(tokens,...)`
  debug (siblings comment these out). Remove.
- **PRE-11 [DEBUG][low]** `item_processor.py` — dozens of commented-out
  `# print(...)` / `# import pdb` lines (`:279-282,327,399-516,571,603-604,...`).
  Strip.
- **PRE-12 [HARDCODE][low]** `item_processor.py:258-262,344-356` action/state
  norm `min/max` arrays + `n_bins=256` hardcoded per class (and drift subtly
  between Action vs ActionState). Source from config.
- **PRE-13 [DEAD][low]** `regenerate_libero_dataset_save_img_action_state_wrist.py`
  imports `json` unused; `preprocess_oft_action_hidden.py:365` orphan
  `del token_count`. Drop.
- **PRE-14 [SMELL][low]** Leftover Chinese dev comments
  (`pretoken_world_model.py:2,18,...`, `pretoken_state_action_model.py:22,93`,
  `action_state_model_conv_generation.py:97` `# TODO: 根据需要统一...`) — noise vs
  the English codebase. `concat_action_world_model_data_libero.py:74-86` has a
  13-line commented-out dead block. Strip.

## Utils / Dataset / Algorithms

- **UDA-01 [SMELL][high]** `utils/wm_image_viz.py:359-631` `visualize_batch()` is a
  273-line method with 3 near-identical decode routes at 4-6 nesting levels;
  `:445-487,528,547,555` has six `except Exception: print(...)` blocks (pollutes
  every DDP rank). Extract `_build_sample_panels(mode,...)` + one `_safe_decode`
  using `logging`.
- **UDA-02 [DUP][med]** `wm_image_viz.py:70-94` `_decode_bpe_block_to_pil` reimplements
  the VQ decode pipeline though `vq_image_decoder.ChameleonImageReconstructor`
  already exists. Route through it.
- **UDA-03 [INCONSISTENT][med]** `utils/torch_utils.py` and `utils/pytorch_util.py`
  are two modules for the same job (`resolve_device/freeze_module/...` vs
  `dict_apply`). Merge `dict_apply` into `torch_utils.py`, drop `pytorch_util.py`
  (2 importers, both vendored chameleon).
- **UDA-04 [INCONSISTENT][high]** Two dataset families: `BaseDataset`
  (HDF5/padding/demo helpers) is inherited by only `PretokenizeDataset`,
  `TokenSequenceDataset`, `PixelSequenceDataset`; `VLASFTHDF5Dataset`,
  `WMReplayClassifierDataset`, `WMPOAlignedLatent{Train,Val}Dataset` extend bare
  `Dataset` and re-implement the same logic. Route all through `BaseDataset`.
- **UDA-05 [DUP][high]** HDF5 open/cache (`swmr=True, libver="latest"`) copy-pasted
  in `pixel_sequence_dataset.py:146`, `pixel_hidden_sequence_dataset.py:281`,
  `vla_sft_hdf5_dataset.py:203`. Add `BaseDataset.cached_hdf5_file(path)`.
- **UDA-06 [DUP][high]** Outcome extraction (`rewards/dones→finish_step→complete`)
  duplicated in `wm_replay_classifier_dataset.py:207-238`,
  `wmpo_aligned_latent_dataset.py:65-92`,
  `collected_rollout_classifier_dataset.py:35-44`. One `_load_demo_outcome(...)`.
- **UDA-07 [DEAD][low]** `base_dataset.py:81 image_from_array()` — 0 callers.
  Remove. `vla_sft_hdf5_dataset.py:43` private `_list_demo_keys()` shadows
  `BaseDataset.list_demo_keys()` — subclass `BaseDataset` instead.
- **UDA-08 [SMELL][high]** `pretokenize_dataset.py` (822 LOC) folds flat-loading +
  windowing + token preprocessing + collation + two classes into one file. Split.
- **UDA-09 [HARDCODE][med]** EOT token `8710` hardcoded ×3 in
  `pretokenize_dataset.py:745,787,788`; Chameleon img/state token IDs
  (`8197/8196/15504/16004`) module-level in `wm_image_viz.py:19-22`. Named
  constants / from tokenizer config.
- **ALG-01 [DUP][high]** PPO clip-loss written 3 ways: `dense.py:355`,
  `dense_chunk.py:312` (`torch.maximum(-adv*ratio,...)`) vs `outcome.py:468-471`
  (`clamp`+`min`). Shared `_ppo_clip_loss(ratio,adv,lo,hi)` in `grpo.py`.
- **ALG-02 [DUP][med]** γ-discount/return + KL-into-reward rebuilt per route
  (`dense.py:297-301`, `dense_chunk.py:260-264`, `outcome.py:368-372`).
  Centralize.
- **ALG-03 [DUP][low]** Entropy-coef lookup diverges: dense uses
  `cfg.get("actent", cfg.get("entropy_coef",0))`; `outcome.py:188` drops the
  `actent` fallback (silent behavioral skew). One `_entropy_coef(cfg)`.
- **ALG-04 [INCONSISTENT][med]** `compute_lambda_returns`
  (`algorithms/dreamervla.py:390-418`) vs `compute_replay_lambda_returns`
  (`:421-453`) differ only in the mask. Merge with an optional flag.
- **ALG-05 [SMELL][high]** `algorithms/dreamervla.py` (1349 LOC)
  `imagine_actor_critic_step` (~`:505-1339`, ~834 lines) does imagination + actor
  + critic/replay + Polyak + metrics in one function. Split.
- **ALG-06 [HARDCODE][med]** Hyperparameters as `.get(...)` defaults in step
  bodies: `dreamervla.py:554 tau=0.02`, `dense.py:87-88 clip 0.2/0.28`,
  `dense.py:121 gamma=1.0`. Validate once into a typed `AlgorithmConfig`
  (matches the repo's "early config validation" rule).
- **UDA-DEBUG [DEBUG][low]** Bare init `print(...,flush=True)` across
  `balanced_terminal_dataset.py:69`, `wmpo_aligned_latent_dataset.py:109,...`,
  `wm_replay_classifier_dataset.py:397,...`, `algorithms/dreamervla.py:1334`. Use
  `logging`.

## Diagnostics / Envs

- **DIAG-01 [DUP][high]** The `torch.load→OmegaConf.create(cfg)→instantiate WM→
  load_state_dict(strict=False)→eval()` block is copy-pasted in ~9-12 scripts
  (`measure_wm_closed_loop.py:146`, `measure_wm_imagine_actor.py:96`,
  `measure_wm_imagine_fidelity.py:82`, `measure_recon_and_action_delta.py:45`,
  `measure_reward_and_drift.py:75`, …); a shared
  `online_utils.load_world_model_state` exists but only 3 use it.
  `eval_chunkwm_closeloop.py:58-96` is a *third* parallel impl. Route all through
  the helper (~−150 LOC; also kills the `ckpt["model"]` vs
  `ckpt["state_dicts"][...]` schema drift).
- **DIAG-02 [DUP][med]** No `add_common_args`: `--device` in 14 files, `--ckpt`
  12, `--seed` 11, `--out*` 18; JSON-summary tail dup in ~18. Add
  `diagnostics/_common.py` (`resolve_device`, `add_ckpt_out_device_args`,
  `write_summary_json`).
- **DIAG-03 [DUP][med]** `slice_latent` redefined in 3 files; `reward_of` in 2;
  `DreamerV3LatentState` reconstruction inlined in 4. Move to
  `utils/latent.py`.
- **DIAG-04 [INCONSISTENT][med]** Device resolution diverges — 7 files guard
  `if torch.cuda.is_available() else "cpu"`; `reward_landscape_sweep.py:142`,
  `finetune_reward_head_sparse.py:153`, `eval_chunkwm_closeloop.py:292` use
  unguarded `torch.device(args.device)`. Single `resolve_device()`.
- **DIAG-05 [HARDCODE][med]** Stale dataset dirname
  `no_noops_t_256_legacy_action_hidden_vla_policy_h2` hardcoded as a default in
  `diagnose_residual_cosine.py:29`, `diagnose_hidden_token_structure.py:34`,
  `eval_chunkwm_closeloop.py:283`. Not an absolute path (passes the hygiene test)
  but a run-specific magic string. Require `--data-dir`.
- **DIAG-06 [DEAD][med]** 16 diagnostics scripts are doc-only one-offs (README row
  only; no code/test/launcher ref). 4 of those are pinned by
  `test_repository_hygiene.py`. Decide as a batch: move under
  `diagnostics/archive/` or prune scripts + README rows + the 4 test pins
  together. (`verify_imports.py` is a useful orphan — add a README row.)
- **DIAG-07 [DEBUG][low]** All diagnostics use bare `print()` (no `logging`).
  Consistent within the suite; optional.
- **ENVS [CLEAN]** `dreamervla/envs/` — no findings (clean inheritance
  eval→train→libero_online, no hardcoded paths, well-sized methods).

## Configs / Repo hygiene

- **CFG-01 [DEAD/HYGIENE][med]** `.planning/` is tracked (10 files: `.active_plan`
  + 3 dated session dirs each with `findings.md`/`progress.md`/`task_plan.md`) —
  per-session agent scratch drafts. `.gitignore` excludes
  `.agents/.claude/.codex/.cursor/` but misses `.planning/`. (Verified: there are
  **no** tracked root-level `findings.md`/`progress.md`/`task_plan.md` — the
  earlier "root scratch files" claim was wrong; the scratch lives under
  `.planning/`.) Fix: `git rm -r --cached .planning/` + add `.planning/` to
  `.gitignore`.
- **CFG-02 [HARDCODE/test-gap][med]** Some tracked docs embedded machine-local
  absolute paths (rooted at `mnt`/`home`) that
  `test_active_files_do_not_pin_machine_local_roots` flags. Fix: use portable
  forms (`~`, `$CONDA_PREFIX`); untracking `.planning/` (CFG-01) removes the
  per-session drafts that carried the rest.
- **CFG-03 [DEAD][high]** `models/chameleon_model/` (see MOD-01) — same item from
  the config/hygiene side; delete + repoint `test_repository_hygiene.py:380`.
- **CFG-04 [DUP][high]** `configs/task/libero_{goal,object,spatial,10}.yaml` — 4
  copy-pasted 117-line files (no `defaults:`) differing in ~8 lines. Extract
  `configs/task/_base_libero.yaml`; each composes `defaults:[_base_libero,_self_]`.
- **CFG-05 [DUP][high]** `configs/dreamervla/rynnvla_wmpo_outcome.yaml` (277L) and
  `openvla_oft_wmpo_outcome.yaml` (241L) — parallel full-size standalones with
  largely duplicated training/optim/`algorithm.wmpo`/critic. Factor
  `_base_wmpo_outcome.yaml`. (Their `_input_token` siblings already compose — leave.)
- **CFG-06 [DUP][med]** `configs/task/OpenVLA_Onetraj_LIBERO{,_10,_Object,_Spatial}.yaml`
  restate suite-suffixed paths (~21-line diffs); template the suffix.
  `configs/classifier/rynnvla_input_token_chunk.yaml` should be a thin
  `defaults:[rynnvla_action_chunk,_self_]` like the openvla_oft pair.
- **CFG-07 [HARDCODE][med]** `model_dim: 4106` literal in 8 configs; it equals the
  `config.py:338`-validated `token_dim(4096)+action_emb_dim(10)*repeat(1)`. Define
  once + interpolate. (X-03)
- **CFG-08 [INCONSISTENT][med]** `configs/task/` mixes `snake_case` and
  `Mixed_Pascal_With_LIBERO` (`OpenVLA_Onetraj_ColdStart_LIBERO_Spatial.yaml`,
  `RynnVLA_LIBERO.yaml`) — the only group with this split. Rename to snake_case,
  update `task=` refs.
- **CFG-09 [DEAD?][med]** `configs/precision/{fp32,fp16,bf16}.yaml`,
  `configs/parallelism/{none,fsdp}.yaml` — composed by zero committed configs
  (CLI-only `+precision=bf16`). Reference from ≥1 smoke recipe or document as
  append-only CLI groups.
- **CFG-10 [HYGIENE][low]** `tutorial_of_git.md` (root, gitignored→untracked,
  stray). `docs/paper/corl/main{,_zh}.pdf` (280K/164K) largest committed binaries
  — fine for a paper repo, flagged.

---

## Verified non-issues (false positives the audits dropped)

These were claimed but disproven on inspection — do **not** "fix" them:

- `algorithms/tdmpc_mpc.py` is **used** by `embodied_eval_runner.py`
  (`TDMPCMPCPlanner`/`TDMPCMPCConfig`) — not dead.
- `MetricLogger` (multi-backend) vs `JsonLogger` (JSONL) are **complementary**
  (both mandated by CLAUDE.md) — not competing; do not "consolidate loggers".
- `online_utils.SuccessTracker` **is** used by `base_runner.py:1019`
  (`console_record_success`) — not dead (may be unused within the online family
  only).
- `preprocess/conversation.py`, `preprocess/paths.py` are intentional re-export
  shims of `utils/` originals — not duplication.
- `_input_token` / `discrete_token` config variants across `worldmodel/`,
  `classifier/`, `dreamervla/`, `VLA/` are already thin `defaults:` overrides —
  large by raw `diff` line-count but correctly factored; leave alone.
- `dreamerv3_torch.py` / `tssm_torch.py` are not vendored upstream copies — they
  house shared WM building blocks.
- The just-added progress reporter (`utils/progress.py`, `console_progress`,
  `format_progress_line`) is intentional and out of scope here.

---

## Suggested execution order

1. **Zero-risk hygiene (one PR):** MOD-01/CFG-03 (delete dead chameleon dup +
   repoint test), CFG-01/CFG-02 (untrack scratch/`.planning`, extend
   `.gitignore`), small dead-symbol removals (UDA-07, MOD-10, PRE-13). Closes the
   machine-path test blind spot.
2. **De-drift constants (one PR):** X-03 (`10004`, `4106`), DIAG-05 dirname,
   UDA-09/ALG-06/PRE-12 magic literals.
3. **Adopt existing shared helpers (per-subsystem PRs):** MOD-02 (`common.py`),
   DIAG-01 (`load_world_model_state`), MOD-04/X-04 (component loader), ALG-01/02
   (PPO helpers), UDA-05/06 (`BaseDataset` caching/outcome).
4. **Structural unification (larger, staged):** RUN-01/02/03 (runners onto base
   infra + one checkpoint scheme), UDA-04 (dataset family), PRE-01 (item_processor
   mixin), RUN-07 (collection base).
5. **Decompose god-files (mechanical, no behavior change):** RUN-11, RUN-12,
   ALG-05, UDA-08.
6. **Maintainer decisions (keep/delete):** PRE-02/PRE-03 (orphan WM scripts +
   unused processors), RUN-14 (multiproc fork), MOD-07 (OFT `official`),
   DIAG-06 (one-off museum), CFG-09 (precision/parallelism groups).

Each numbered item carries its own `file:line` so the maintainer can accept,
defer, or reject it individually.
