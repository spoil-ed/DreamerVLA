# Token-WMPO Classifier Design

Date: 2026-07-07
Status: approved for implementation after audit

## Goal

Validate a WMPO-style success classifier while keeping the input modality aligned
with DreamerVLA's OpenVLA-OFT token-hidden pipeline rather than pixel VideoMAE.

This is phase A only:

- Train and validate a classifier checkpoint.
- Report window-level and episode-level metrics.
- Save the best operating threshold in the checkpoint.
- Do not wire this checkpoint into cotrain or policy RL yet.

## Scope

The target is a tokenized adaptation of WMPO's reward-model protocol:

- Successful episode terminal window: positive.
- Successful episode earlier window: negative.
- Failed episode terminal window: negative.
- Failed episode earlier window: negative.
- Episode inference: slide windows over the trajectory and classify the episode
  as successful when any window exceeds the selected threshold.

The input is OpenVLA-OFT input-token hidden state, not RGB:

- Per-frame token grid: `token_count x token_dim`.
- Classifier window: `window x token_count x token_dim`, with optional proprio
  and language conditioning already supported by the current model.
- Model: `LatentSuccessClassifier` with `head_type=spatial_tf`.

## Audit Findings

Existing code already covers most of the protocol:

- `LumosAlignedLatentTrainDataset` and `LumosAlignedLatentValDataset` implement
  the same terminal/earlier-window label scheme as the WMPO reference reward
  model.
- `_load_demo` derives `finish_step` from `dones` and `complete` from
  `sparse_rewards` or `rewards`, which matches collected-rollout and processed
  LIBERO data.
- `LatentSuccessClassifier.head_type=spatial_tf` accepts token-grid inputs or
  flat token-grid inputs and keeps spatial token structure before classification.
- `LatentClassifierRunner._evaluate_episode_level` already implements the
  sliding-window any-positive episode protocol.
- `_save_named` stores `model`, `threshold`, `f1`, `step`, and classifier config,
  which is sufficient for later cotrain integration.

Gaps to close before validation:

- The existing experiment name says `latent_classifier...`; add a dedicated
  `wmpo_token_classifier...` experiment so results are clearly attributed.
- Episode-level evaluation is disabled in the current config; enable it for this
  validation route.
- Diagnostics need to make the phase-A decision obvious: train/val positive and
  negative counts, best threshold, confusion counts, probability distribution,
  and saved checkpoint path.
- The dataset docstring still mentions a specific hidden source; update wording
  to say token/action-hidden sidecars.

## Design

Add a new experiment config:

`configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml`

It will reuse:

- `_target_: dreamervla.runners.LatentClassifierRunner`
- `LumosAlignedLatentTrainDataset`
- `LumosAlignedLatentValDataset`
- `LatentSuccessClassifier(head_type=spatial_tf)`

Recommended defaults:

- `training.out_dir`: `data/outputs/classifier/wmpo_token_cls_openvla_onetraj_libero_goal_h1/...`
- `training.batch_size`: keep small enough for token-grid memory, initially `4`
  if GPU memory allows, otherwise `1`.
- `training.lr`: `3.0e-5`, matching the recent stable classifier run.
- `training.num_epochs`: `8` for the full validation run.
- `training.eval_every`: `250`.
- `training.episode_eval_enabled`: `true`.
- `data.window`: `8`.
- `data.stride_train`: `8`.
- `data.stride_val`: `8` for window validation.
- `training.episode_eval_stride`: `1` for WMPO-style episode scoring.
- `data.chunk_subsample`: OpenVLA-OFT action chunk size.
- `data.chunk_pool`: `last`.

Enhance metrics without changing classifier semantics:

- Extend threshold sweep output with TP/TN/FP/FN and predicted positive/negative
  counts.
- Log a dataset summary event during setup with demo counts and per-epoch
  positive/negative window counts.
- Include best checkpoint path in the final summary when available.

## Validation

Static and unit validation:

- Compose the new Hydra experiment.
- Unit-test that the new experiment resolves to `spatial_tf`, token count,
  chunk granularity, and episode eval enabled.
- Unit-test the extended threshold metrics confusion counts.

Runtime validation:

- Run a short smoke train/eval with a small step budget to verify data loading,
  token shapes, logging, and checkpoint save.
- Run the full phase-A classifier training if the smoke passes.
- Accept phase A if window-level and episode-level F1 are both non-zero and the
  checkpoint contains a threshold plus classifier config.

## Non-Goals

- No pixel VideoMAE model in this phase.
- No cotrain/RL reward integration in this phase.
- No changes to actor update, world model, or rollout collection.
- No destructive cleanup of existing classifier checkpoints.
