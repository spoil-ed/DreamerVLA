# Parameter reference

The single reference for every adjustable parameter. All training is the grouped
Hydra entry `python -m dreamervla.train experiment=<name> task=<suite>`; everything
below is a `key=value` override (or a shell-launcher convenience key). Tutorials:
[`docs/experiment_tutorials/`](experiment_tutorials/); rationale:
[`docs/experiment_tutorials/EXPLAINED.md`](experiment_tutorials/EXPLAINED.md).

## Convenience overrides (shell launchers)

`scripts/train_vla.sh`, `train_wm.sh`, `train_dreamervla.sh`, `eval_libero_vla.sh`
accept these before any `--` and map them to Hydra keys / env vars:

| Launcher key | Maps to | Default | Meaning |
| --- | --- | --- | --- |
| `gpus` | `CUDA_VISIBLE_DEVICES` | none | comma-separated GPU ids, e.g. `0,1,2,3` |
| `ngpu` | nproc / world size | inferred from `gpus` | number of GPUs |
| `batch_size` | `dataloader.batch_size` (or `training.batch_size`) | per-config | local batch per GPU |
| `num_workers` | `dataloader.num_workers` (or `training.num_workers`) | per-config | dataloader workers |
| `num_epochs` | `training.num_epochs` | 20 | training epochs |
| `max_steps` | `training.max_train_steps` | null | step cap (overrides epochs) |
| `out_dir` | `OUT_DIR` → `training.out_dir` | timestamped | run output root |

Anything after `--` is passed verbatim as Hydra overrides (dotted keys).

## Config groups

`experiment=` selects a recipe; the other groups are pulled in by the experiment or
overridable directly.

| Group | Purpose |
| --- | --- |
| `experiment/` | top-level recipe (VLA SFT / world-model / classifier / DreamerVLA / eval / online cotrain / collect) |
| `VLA/` | VLA backbone + finetune strategy (rynnvla / openvla_oft, full / one-trajectory) |
| `worldmodel/` | DINO-WM architecture + input mode (action-chunk / input-token / discrete-token) |
| `classifier/` | latent success classifier (outcome-reward LUMOS) |
| `dreamervla/` | actor-critic / LUMOS training (all inherit `_base_lumos`) |
| `evaluation/` | LIBERO rollout eval (`libero_vla`) |
| `task/` | LIBERO suite + dataset/checkpoint paths (snake_case `task=` token) |
| `logger/` | `tensorboard` / `wandb` / `tensorboard_wandb` |

## `training.*` — training loop

| Key | Default | Meaning |
| --- | --- | --- |
| `training.out_dir` | timestamped under `data/outputs/` | checkpoints + logs + resolved config |
| `training.device` | `cuda` | CUDA device |
| `training.debug` | `false` | tiny smoke instead of the full run |
| `training.resume` | `false` | resume from checkpoint |
| `training.distributed_strategy` | `ddp` | `ddp` / `fsdp` / `single` |
| `training.fsdp_mixed_precision` | `bf16` | `fp32` / `fp16` / `bf16` |
| `training.enable_activation_checkpointing` | `true` | gradient checkpointing (memory) |
| `training.num_epochs` | `20` | epochs |
| `training.max_train_steps` | `null` | step cap (overrides epochs) |
| `training.gradient_accumulate_every` | `1` | grad-accum steps |
| `training.checkpoint_every` | `1` (DreamerVLA, epochs) / `500` (VLA, steps) | save cadence |
| `training.lr_scheduler` | `cosine` (DreamerVLA) / `constant` (OFT VLA) | LR schedule |
| `training.lr_warmup_steps` | `500` (DreamerVLA) | warmup steps |
| `training.run_wm_phase` | `true` | run the WM supervised phase (DreamerVLA) |
| `training.run_actor_critic_phase` | `true` | run the actor-critic / LUMOS phase (DreamerVLA) |
| `training.use_ema` | `false` | EMA of weights |

## `dataloader.*` — data loading

| Key | Default | Meaning |
| --- | --- | --- |
| `dataloader.batch_size` | `2`–`16` per route | local batch per GPU |
| `dataloader.num_workers` | `2`–`4` | worker processes |
| `dataloader.shuffle` | `true` | shuffle each epoch |
| `dataloader.drop_last` | `true` (DreamerVLA/WM) | drop the trailing partial batch |
| `dataloader.pin_memory` | `false` (DreamerVLA) / `true` (VLA/WM) | pin tensors |
| `dataloader.persistent_workers` | `true` | keep workers between epochs |
| `dataloader.prefetch_factor` | `1` (DreamerVLA) / `2` (WM) | batches prefetched per worker |
| `dataloader.multiprocessing_context` | `forkserver` (DreamerVLA) / `null` (WM) | `fork`/`spawn`/`forkserver`/`null` |

## `dataset.*` — windowing (BalancedTerminalDataset)

| Key | Default | Meaning |
| --- | --- | --- |
| `dataset.hdf5_dir` / `dataset.hidden_dir` | `${task.*}` | filtered HDF5 + latent cache |
| `dataset.reward_mode` | `per_window_dense` | reward labeling |
| `dataset.balanced_length` | `50000` | resampled window count |
| `dataset.sequence_length` | `24`/`36` (= `H + N*K + 1`) | context window |
| `dataset.stride` | `1` | window stride |
| `dataset.max_files` / `max_demos_per_file` / `max_windows` | `null` | smoke-run caps |

## `world_model.*` — DINO-WM chunk predictor

`obs_dim = time_horizon × action_dim × token_dim`. `model_dim` and the dims below must
match the WM checkpoint being loaded.

| Key | RynnVLA | OpenVLA-OFT | Meaning |
| --- | --- | --- | --- |
| `world_model.chunk_size` | 5/10 | 8 | action chunk K (env-steps) |
| `world_model.token_count` | 35 | 56 | VLA tokens per frame |
| `world_model.token_dim` | 1024 | 4096 | VLA token embedding dim |
| `world_model.model_dim` | **1034** | **4106** | transformer model dim |
| `world_model.depth` / `heads` / `dim_head` / `mlp_dim` | 6 / 16 / 64 / 2048 | 6 / 16 / 256 / 4096 | transformer size |
| `world_model.num_hist` | 3 | 3 | autoregressive history (required = 3) |
| `world_model.latent_stage` | `query_after` | `query_after`/`query_before` | extraction point (Scheme A vs 1) |
| `world_model.reward_head_type` | `binary` | `binary` | reward head |
| `world_model.reward_loss_scale` | 32.0 | 32.0 | reward loss weight |
| `world_model.chunk_rollout_chunks` | 4 | 4 | anti-drift rollout depth |
| `world_model.freeze_backbone` | `true` | `true` | freeze the DINO image backbone |

## `algorithm.*` — RL (LUMOS / actor-critic)

| Key | Default | Meaning |
| --- | --- | --- |
| `algorithm.update_type` | `LUMOS` | route: `LUMOS` / `dreamer` / others in the registry |
| `algorithm.lumos.episode_max_steps` | `300` | imagined env-steps per outcome rollout (LIBERO horizon) |
| `algorithm.lumos.chunk_size` | `${task...chunk_size}` | K; must equal `world_model.chunk_size` and `policy.time_horizon` |
| `algorithm.lumos.classifier_min_steps` | 3 (OFT) / 4 (RynnVLA) | min chunk index for the classifier sweep |
| `algorithm.lumos.filter_zero_variance_groups` | `true` | skip GRPO groups with no within-group variance |
| `algorithm.imag_last` | `4` | replay start states per window (a memory dial) |
| `algorithm.imagination_horizon` | `5` | imagined frames |
| `algorithm.lam` | `0.95` | λ-return trace |
| `algorithm.ppo_gamma` | `1.0` | discount (1.0 = episodic) |
| `algorithm.actent` | `0.0` | entropy bonus (read via `actent` → `entropy_coef`) |
| `algorithm.kl_coef` | `0.01` | KL-to-ref reward penalty |
| `algorithm.actor_bc_to_ref_scale` | `0.1` | BC-to-reference anchor weight |
| `algorithm.ppo_rollouts_per_start` | `4` | GRPO rollouts per start (a memory dial) |
| `algorithm.ppo_update_epochs` | `1` | PPO epochs per batch |
| `algorithm.clip_ratio_low` / `clip_ratio_high` | `0.2` / `0.28` | PPO clip bounds |
| `algorithm.clip_ratio_c` | `3.0` | RLinf dual-clip constant (no-op when null) |
| `algorithm.clip_log_ratio` | `10.0` | log-ratio clamp before `exp` (no-op when null) |
| `algorithm.advantage_eps` | `1.0e-6` | advantage normalization epsilon |
| `algorithm.repval_loss` | `true` | replay value loss (bootstraps with the critic value — A4) |
| `algorithm.repval_scale` | `0.3` | replay value loss weight |
| `algorithm.repl_loss.slowtar` | `false` | use the target critic for the replay bootstrap |
| `algorithm.slowtar` / `slowreg` / `target_critic_tau` | `false` / `1.0` / `0.02` | critic target controls |
| `algorithm.return_normalization.mode` | `dreamerv3` | return normalization (`none` to disable) |

`B_eff = dataloader.batch_size × algorithm.imag_last × algorithm.ppo_rollouts_per_start`
is the online-cotrain memory dial — see EXPLAINED.md.

## `optim.*` — optimizers (per module)

| Key | Default | Meaning |
| --- | --- | --- |
| `optim.grad_clip_norm` | `1.0` | global grad-norm clip |
| `optim.world_model.{name,lr,betas,eps,weight_decay}` | adamw, 2e-6, [0.9,0.999], 1e-20, 0.0 | WM optimizer |
| `optim.policy.{...}` | adam, 5e-7, [0.9,0.95], 1e-8, 0.0 | actor optimizer |
| `optim.critic.{...}` | adam, 3e-4, [0.9,0.95], 1e-8, 0.01 | critic optimizer |

## `critic.*` — two-hot critic

`_target_` is per-route (OFT uses `${task.openvla_oft.critic_target}`). Body:
`hidden_dim 1024`, `critic_layers 3`, `activation silu`, `norm rms`, `num_bins 255`,
`bin_min -20.0`, `bin_max 20.0`.

## `policy.*` — actor

`_target_` per route (`RynnVLAActionHiddenActor` / `LatentToOpenVLADiscreteTokenActor`
/ `${task.openvla_oft.actor_target}`). Common: `action_dim 7`, `adapter_type
residual_mlp`, `adapter_hidden_dim 1024`, `initial_log_std -3.0`, `min/max_log_std
-5.0/-2.0`, `time_horizon = K`.

## `init.*` — checkpoint init

| Key | Default | Meaning |
| --- | --- | --- |
| `init.vla_ckpt_path` | `${task.vla_ckpt_path}` / `${task.openvla_oft.ckpt_path}` | VLA backbone ckpt |
| `init.world_model_state_ckpt` | `null` (OFT) / path (RynnVLA) | WM weights |
| `init.classifier_state_ckpt` | `null` (OFT) / path (RynnVLA) | outcome classifier |
| `init.dreamervla_state_ckpt` | `null` | full DreamerVLA ckpt |
| `init.load_dreamervla_{world_model,critic,target_critic,policy,return_tracker}` | `false` | selective loads from the full ckpt |
| `init.load_dreamervla_strict` | `false` | strict load |
| `init.reset_world_model_reward_head` | `false` | re-init reward head after load |

## `eval.*` — LIBERO rollout (`evaluation=libero_vla`)

| Key | Default | Meaning |
| --- | --- | --- |
| `eval.ckpt_path` | `null` | checkpoint to evaluate |
| `eval.ckpt_kind` | `auto` | `auto` / `vla` / `dreamer` |
| `eval.task_suite_name` | `libero_goal` | LIBERO suite |
| `eval.num_episodes_per_task` | `50` | rollouts per task |
| `eval.num_steps_wait` | `10` | settle steps before first action |
| `eval.dreamer_policy_source` | `ckpt` | `ckpt` / `env` |
| `eval.dreamer_actor_input_source` | `rssm` | actor input (`rssm` / `encoder`) |
| `eval.target_token_id` | `10004` (= `DEFAULT_ACTION_TOKEN_ID`) | action-token id; see [`dreamervla/constants.py`](../dreamervla/constants.py) |
| `eval.save_video` | `false` | dump rollout videos |
| `eval.tdmpc_mpc.enabled` | `false` | optional TDMPC planner |

## `logger=` — logging

`logger=tensorboard_wandb` (default) / `tensorboard` / `wandb`. Knobs:
`runner.logger.wandb_mode` (`online`/`offline`/`disabled`),
`runner.logger.project_name` (`dreamervla`), `runner.logger.log_path`
(`${training.out_dir}/log`). See EXPLAINED.md for TensorBoard/W&B viewing.

## Constants

`dreamervla/constants.py`: `DEFAULT_ACTION_TOKEN_ID = 10004` — the OpenVLA-OFT /
Chameleon action-token start id (a vocab constant; override the per-route
`target_token_id` only for a different-vocab backbone).

## Environment variables

| Var | Purpose |
| --- | --- |
| `DVLA_ROOT` / `DVLA_DATA_ROOT` | project root / data root |
| `OUT_DIR` | output dir (set by `out_dir=`) |
| `CUDA_VISIBLE_DEVICES` | GPUs (set by `gpus=`) |
| `MUJOCO_GL=osmesa` | LIBERO rendering backend (EGL crashes on some hosts) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduce OOM in online cotrain |
| `NCCL_NVLS_ENABLE=0` | avoid NVLink-SHARP DDP hangs |

## Key interdependencies

- `world_model.chunk_size` == `algorithm.lumos.chunk_size` == `policy.time_horizon` (= K).
- `world_model.obs_dim` = `time_horizon × action_dim × token_dim`.
- `world_model.model_dim` must match the WM checkpoint loaded via `init.world_model_state_ckpt`.
- `dataset.sequence_length` = `H + N*K + 1` (H imagination horizon, N rollout chunks).
