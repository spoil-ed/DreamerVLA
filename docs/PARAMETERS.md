# Parameter Reference

All training runs through the grouped Hydra entry:

```bash
python -m dreamervla.train experiment=<name> task=<suite>
```

Shell launchers expose a small set of convenience keys and pass remaining
`key=value` arguments directly to Hydra.

## Convenience Overrides

| Launcher key | Maps to | Default | Meaning |
| --- | --- | --- | --- |
| `gpus` | `CUDA_VISIBLE_DEVICES` | none | comma-separated GPU ids |
| `ngpu` | nproc / world size | inferred from `gpus` | number of GPUs |
| `batch_size` | `dataloader.batch_size` | per-config | local batch per GPU |
| `num_workers` | `dataloader.num_workers` | per-config | dataloader workers |
| `out_dir` | `training.out_dir` | timestamped | run output root |
| `--resume PATH` | `training.resume*` | disabled | resume a run/checkpoint in its original run root |

## Config Groups

| Group | Purpose |
| --- | --- |
| `experiment/` | top-level recipes for collect, warmup, cotrain, eval |
| `dreamervla/` | cotrain runner recipes |
| `worldmodel/` | world-model construction and Ray worker adapter |
| `classifier/` | classifier construction and Ray worker adapter |
| `evaluation/` | LIBERO rollout eval |
| `task/` | LIBERO suite + checkpoint + sidecar metadata |
| `logger/` | `tensorboard`, `wandb`, `tensorboard_wandb` |

## Warmup and Cotrain

| Key | Meaning |
| --- | --- |
| `offline_warmup.data_dir` | reward-HDF5 replay directory |
| `offline_warmup.hidden_dir` | hidden-sidecar replay directory |
| `offline_warmup.task_id` | optional single-task replay filter |
| `training.wm_warmup_steps` | world-model update budget |
| `training.classifier_warmup_steps` | classifier update budget |
| `training.warmup_replay_epochs` | replay-pass derived update budget |
| `training.warmup_checkpoint_every` | warmup checkpoint cadence |
| `training.wm_profile_steps` | bounded WM update profile budget; `-1` is diagnostic-only all-step profiling |
| `training.wm_prefetch_workers` | CPU replay batches built ahead of the current WM update |
| `training.world_model_ddp.*` | opt-in WM-only DDP flags; the offline fixed-graph recipe enables `static_graph` and `gradient_as_bucket_view` |
| `training.update_profile_steps` | bounded standalone classifier update profile budget |
| `training.precision` | standalone classifier autocast precision (`bf16` on H100) |
| `training.classifier_batch_size` | local classifier batch |
| `dataloader.batch_size` | local WM replay batch |
| `online_rollout.sequence_length` | replay window length |
| `online_rollout.total_env_steps` | online cotrain budget |
| `online_rollout.num_envs` | vector env count for sync path |
| `online_rollout.buffer_size` | replay capacity |

### Manual Ray failure-conditioned imagined RL

| Key | Meaning |
| --- | --- |
| `manual_cotrain.training_mode` | active mainline is `failure_imagined_rl` |
| `manual_cotrain.initial_condition_selector` | active selector is `failed_episode_start`; future selectors remain config extensions |
| `manual_cotrain.staged_policy_update` | legacy full-cotrain barrier switch; `false` in imagined-only mode |
| `manual_cotrain.learner_updates_enabled` | WM/CLS optimizer switch; `false` in imagined-only mode |
| `manual_cotrain.real_rollout_target_trajectories` | exact completed real trajectories drained per global step; mainline is `32` |
| `manual_cotrain.max_policy_kl` | per-step actor PPO trust-region allowance |
| `manual_cotrain.wm_rollout_target_trajectories` | imagined trajectories used by actor PPO |
| `manual_cotrain.wm_env_write_replay` | whether imagined episodes enter replay; mainline keeps this `false` |
| `manual_cotrain.save_replay_state` | persist the historical failed-episode pool for exact resume |
| `manual_cotrain.checkpoint_every` | completed global-step checkpoint cadence; segmented eval forces its boundary checkpoint |
| `training.resume` | enable restoration for the selected runner |
| `training.resume_path` | exact checkpoint resolved by the launcher |
| `training.resume_dir` | owning run root reused by the resumed invocation |
| `manual_cotrain.resume_ckpt` | legacy/internal full-checkpoint override; public launchers should use `--resume` |
| `actor.train_cfg.optimizers.policy.lr` | original LM/OFT actor PPO LR |

The canonical checkpoint tree is `${training.out_dir}/checkpoints/`. Public train
launchers accept either a run root or a checkpoint:

```bash
bash scripts/experiments/world_model_training/train.sh \
  --config dreamer-wm --resume /path/to/dreamer-wm/20260714_120000

bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero --resume /path/to/openvla_libero/20260714_120000
```

`--resume` cannot be combined with a new `out_dir`; it intentionally appends logs and
checkpoints to the original run root.

Cotrain real-rollout and evaluation progress bars use completed trajectories as the
numerator and the configured trajectory total as the denominator. `chunks` remains a
diagnostic status field and never advances the primary bar.

## OpenVLA-OFT Token Contract

The current cotrain path reads token metadata from
`task.openvla_oft.hidden_token.*`.

| Key | Meaning |
| --- | --- |
| `expected_action_head_type` | action-head contract stored in sidecar metadata |
| `expected_obs_hidden_source` | projected hidden-token source expected by WM/classifier |
| `expected_prompt_style` | prompt serialization contract |
| `expected_history` | image/history count encoded by the sidecar |
| `expected_include_state` | whether VLA-side state was included |
| `expected_rotate_images_180` | image-rotation contract |
| `token_count` | projected visual-token count per frame; checkpoint/sidecar derived, currently `256` |
| `token_dim` | projected visual-token width; checkpoint/sidecar derived, currently `4096` |
| `wm_obs_dim` | flattened WM observation dim |
| `chunk_size` | action chunk size |
| `proprio_dim` | replay proprio dimension |
| `model_dim` | WM transformer width after feature assembly |

## World Model

| Key | Meaning |
| --- | --- |
| `world_model.chunk_size` | action chunk K |
| `world_model.token_count` | VLA tokens per frame |
| `world_model.token_dim` | VLA token embedding dim |
| `world_model.model_dim` | transformer model dim |
| `world_model.depth` / `heads` / `dim_head` / `mlp_dim` | transformer size |
| `world_model.num_hist` | autoregressive history |
| `world_model.latent_stage` | extraction point |
| `world_model.reward_head_type` | reward head |
| `world_model.reward_loss_scale` | reward loss weight |
| `world_model.chunk_rollout_chunks` | rollout-loss depth |
| `world_model.chunk_rollout_loss_scale` | rollout-loss weight |
| `world_model.cosine_loss_scale` | optional cosine contribution to the WM train loss; cosine is reported even when its weight is zero |
| `world_model.proprio_reconstruction_loss_scale` | proprio reconstruction weight |

## Optimizers

| Key | Meaning |
| --- | --- |
| `optim.param_precision` | WM master-parameter and optimizer-state precision (`fp32` or `bf16`) |
| `optim.precision` | WM forward/backward autocast precision (`fp32`, `bf16`, or `fp16`) |
| `optim.grad_clip_norm` | global grad-norm clip |
| `optim.world_model.lr` | WM learning rate |
| `optim.world_model.betas` | WM Adam betas |
| `optim.world_model.eps` | WM Adam/AdamW numerical epsilon |
| `optim.world_model.weight_decay` | WM weight decay |
| `optim.policy.lr` | actor learning rate |
| `optim.critic.lr` | critic/classifier learning rate |

The Ray learner uses the corresponding
`learner.train_cfg.param_precision`, `learner.train_cfg.precision`, and
`learner.train_cfg.optimizers.<component>.*` keys. The Dreamer-WM recipe keeps
parameters and AdamW moments in FP32 while using BF16 autocast; DINO-WM keeps
both controls at FP32.

## Evaluation

| Key | Meaning |
| --- | --- |
| `eval.ckpt_path` | checkpoint to evaluate |
| `eval.ckpt_kind` | `auto`, base `vla`, complete learned `vla_policy`, or legacy `dreamer` |
| `eval.task_suite_name` | LIBERO suite |
| `eval.num_episodes_per_task` | rollouts per task |
| `eval.num_steps_wait` | settle steps before first action |
| `eval.save_video` | dump rollout videos |
| `eval.cotrain_diagnostics` | attach read-only WM/CLS causal diagnostics to a `vla_policy` eval |
| `eval.cotrain_expected_trajectories` | exact trajectory count required before diagnostics are accepted; mainline fixed protocol is `100` |
| `eval.cotrain_encode_batch_size` | raw-policy encoding batch used by the streaming diagnostic observer |

For the staged mainline, diagnostic trajectories never enter replay, optimizers, or
classifier threshold calibration. WM evaluation seeds only its configured real history
and then recursively carries the model-returned hidden/action histories across every
complete action chunk.

## Environment Variables

| Var | Purpose |
| --- | --- |
| `DVLA_ROOT` / `DVLA_DATA_ROOT` | project root / data root |
| `CUDA_VISIBLE_DEVICES` | visible GPUs |
| `NGPU` | torchrun process count for shell launchers |
| `MUJOCO_GL=osmesa` | LIBERO rendering backend |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduce allocator fragmentation |
| `NCCL_NVLS_ENABLE=0` | avoid NVLink-SHARP DDP hangs on affected hosts |

## Key Interdependencies

- `world_model.chunk_size`, `algorithm.lumos.chunk_size`, and
  `policy.time_horizon` must match.
- `world_model.obs_dim` is derived from token count and token dim.
- `world_model.model_dim` must match the checkpoint being loaded.
- `dataset.sequence_length` must cover history, rollout chunks, and the next
  target frame.
