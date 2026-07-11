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

## Config Groups

| Group | Purpose |
| --- | --- |
| `experiment/` | top-level recipes for collect, warmup, cotrain, eval |
| `dreamervla/` | cotrain runner recipes |
| `evaluation/` | LIBERO rollout eval |
| `task/` | LIBERO suite + checkpoint + sidecar metadata |
| `logger/` | `tensorboard`, `wandb`, `tensorboard_wandb` |
| `precision/` | manual learner AMP precision |
| `parallelism/` | manual learner sharding knobs |
| `scheduler/` | single-node Ray placement metadata |

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
| `training.update_profile_steps` | bounded standalone classifier update profile budget |
| `training.precision` | standalone classifier autocast precision (`bf16` on H100) |
| `training.classifier_batch_size` | local classifier batch |
| `dataloader.batch_size` | local WM replay batch |
| `online_rollout.sequence_length` | replay window length |
| `online_rollout.total_env_steps` | online cotrain budget |
| `online_rollout.num_envs` | vector env count for sync path |
| `online_rollout.buffer_size` | replay capacity |

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
| `token_count` | token count per frame |
| `token_dim` | token embedding dim |
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
| `world_model.proprio_reconstruction_loss_scale` | proprio reconstruction weight |

## Optimizers

| Key | Meaning |
| --- | --- |
| `optim.grad_clip_norm` | global grad-norm clip |
| `optim.world_model.lr` | WM learning rate |
| `optim.world_model.betas` | WM Adam betas |
| `optim.world_model.weight_decay` | WM weight decay |
| `optim.policy.lr` | actor learning rate |
| `optim.critic.lr` | critic/classifier learning rate |

## Evaluation

| Key | Meaning |
| --- | --- |
| `eval.ckpt_path` | checkpoint to evaluate |
| `eval.ckpt_kind` | `auto`, `vla`, or `dreamer` |
| `eval.task_suite_name` | LIBERO suite |
| `eval.num_episodes_per_task` | rollouts per task |
| `eval.num_steps_wait` | settle steps before first action |
| `eval.save_video` | dump rollout videos |

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
