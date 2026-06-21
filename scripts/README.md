# Script Registry

`scripts/` contains resumable shell launchers only. Python implementation code
lives under the `dreamervla` package and is launched with `python -m`.

## Main Path

| Script | Purpose |
| --- | --- |
| `install_env.sh` | Hydra wrapper for resumable install steps under `scripts/install/` |
| `download_assets.sh` | Hydra wrapper for selected download steps under `scripts/download/` |
| `preprocess_libero.sh` | Hydra wrapper that preprocesses the standard LIBERO suites |
| `preprocess/prepare_libero_data.sh` | Hydra wrapper for one LIBERO suite preprocessing chain |
| `train_vla.sh` | VLA SFT via Hydra experiment configs |
| `train_wm.sh` | World-model and classifier training via Hydra experiment configs |
| `train_dreamervla.sh` | DreamerVLA training via Hydra experiment configs |
| `collect_parallel.sh` | Data-parallel Ray cold-start collection (one job per GPU), merged into one dir for the warmup/cotrain launcher (`skip_collect=true`) |
| `e2e_coldstart_warmup_cotrain_ray.sh` | Ray cold-start collection followed by offline WM/classifier warmup |
| `e2e_coldstart_warmup_cotrain_noray.sh` | Pure-Hydra cold-start collection followed by offline WM/classifier warmup |
| `eval_libero_vla.sh` | LIBERO rollout eval for VLA or Dreamer checkpoints |
| `start_ray.sh` | Start a local single-node Ray head for manual backend debugging |
| `check_ray.sh` | Inspect the active Ray cluster status |

## Install Steps

| Script | Purpose |
| --- | --- |
| `install/00_apt_tools.sh` | System packages |
| `install/10_conda_env.sh` | Conda environment |
| `install/20_torch.sh` | PyTorch CUDA wheel set |
| `install/30_python_deps.sh` | Python runtime and dev dependencies |
| `install/40_third_party.sh` | LIBERO, robosuite stack, OpenSora, and OpenVLA-OFT third-party packages |
| `install/50_special_packages.sh` | flash-attn, egl_probe, and optional apex / TensorNVMe |
| `install/60_verify.sh` | Import and CUDA visibility check |

## Download Steps

| Script | Purpose |
| --- | --- |
| `download/10_rynnvla.sh` | Download RynnVLA Chameleon, Lumina, VLA, and action-WM weights |
| `download/20_openvla_oft.sh` | Download OpenVLA-OFT HDF5 SFT checkpoints from user-provided repos |
| `download/30_openvla_oft_one_trajectory.sh` | Download OpenVLA-OFT one-trajectory checkpoints |
| `download/40_libero_dataset.sh` | Download LIBERO suites into `datasets/libero/<suite>/` |
| `download/50_calvin_dataset.sh` | Download CALVIN tasks into `datasets/calvin/`; supports official, Hugging Face mirror, and OpenDataLab methods |

Download steps are intentionally serial and numbered. To add a new asset
family, create `download/NN_name.sh`, write outputs only under
`${DVLA_DATA_ROOT}`, then append the new script to
`configs/scripts/download.yaml`.

## Preprocessing

| Script | Purpose |
| --- | --- |
| `preprocess_libero.sh` | Top-level wrapper around `preprocess/prepare_libero_data.sh` for one or more LIBERO suites |
| `preprocess/prepare_libero_data.sh` | Run the standard RynnVLA-002 LIBERO preprocessing chain |
| `preprocess/process_all_libero_data.sh` | Compatibility wrapper for the pretokenized-dataset step across suites |
| `preprocess/10_hdf5_reward.sh` | Write LIBERO config, mark/filter HDF5 files, and add reward labels |
| `preprocess/20_pretokenize_dataset.sh` | Build image/state trees, conv JSONs, token records, manifests, and YAML configs |
| `preprocess/30_action_hidden.sh` | Extract legacy RynnVLA action-hidden sidecars |
| `preprocess/35_oft_action_hidden.sh` | Extract OpenVLA-OFT Scheme-A action-hidden sidecars (L1 or discrete checkpoints) |
| `preprocess/40_validate.sh` | Validate generated LIBERO preprocessing artifacts |
| `preprocess/validate_libero_data.sh` | Fast structural validation for LIBERO preprocessing outputs |
| `preprocess/concat_record_libero.sh` | Concatenate LIBERO record files |

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamervla.launchers.workflow` | Hydra workflow runner for install/download/preprocess shell steps |
| `dreamervla.preprocess.filter_marked_libero_hdf5` | Filter no-op marked HDF5 files |
| `dreamervla.preprocess.preprocess_remaining_steps_reward` | Remaining-steps reward labels |
| `dreamervla.preprocess.validate_libero_data_prep` | Structural validation for HDF5, conv, token, record, manifest, and config counts |
| `dreamervla.preprocess.preprocess_rynn_pixel_hidden` | RynnVLA action-hidden sidecar extraction |
| `dreamervla.preprocess.preprocess_oft_action_hidden` | OpenVLA-OFT action-hidden sidecar extraction |

Common launcher flags stay intentionally small:

    bash scripts/install_env.sh only=[20_torch] force=true
    bash scripts/download_assets.sh download.rynnvla=false download.libero=true env.LIBERO_SUITES=libero_goal
    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1 num_procs=8

    bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
      gpus=0,1 ngpu=2 batch_size=16 num_workers=4 num_epochs=20

LIBERO preprocessing GPU and worker controls:

- `gpus=0` selects visible GPUs and is passed through as `CUDA_VISIBLE_DEVICES`.
- `ngpu=1` controls the action-hidden extraction `torchrun --nproc-per-node`
  count. For multi-GPU extraction, keep it aligned with the number of selected
  GPUs.
- `num_procs=8` controls CPU worker processes for pretokenization. It is not
  the same as training `num_workers`.

Single-suite, single-GPU preprocessing:

    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal \
      gpus=0 ngpu=1 num_procs=8

Single-suite, multi-GPU preprocessing:

    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal \
      gpus=0,1 ngpu=2 num_procs=16

Run only the CPU pretokenization step:

    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal \
      only='[20_pretokenize_dataset]' gpus=0 num_procs=8

Run only the RynnVLA action-hidden extraction step:

    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal \
      only='[30_action_hidden]' gpus=0,1 ngpu=2

Process multiple LIBERO suites:

    LIBERO_SUITES="libero_goal libero_object libero_spatial libero_10" \
    bash scripts/preprocess_libero.sh gpus=0,1 ngpu=2 num_procs=16

Training launchers are Hydra wrappers. `experiment=...` selects a config group under
`configs/experiment/`; `task=...`, `gpus=...`, `ngpu=...`, `batch_size=...`,
and `num_workers=...` are script-level overrides; any other `key=value`
argument is passed to the real training config unchanged.
Grouped training defaults to `logger=tensorboard_wandb`, so each run writes
local TensorBoard events under `${training.out_dir}/log/tensorboard` and W&B run
files under `${training.out_dir}/log/wandb`. W&B defaults to online mode; add
`runner.logger.wandb_mode=offline` for local-only W&B logs. Use
`logger=tensorboard` or `logger=wandb` only when you want a single backend.

To view TensorBoard metrics, point TensorBoard at the run's log directory:

    tensorboard --logdir "${OUT_DIR}/log/tensorboard" --host 0.0.0.0 --port 6006

On a remote training host, forward the port and open `http://localhost:6006`:

    ssh -L 6006:localhost:6006 user@host

For W&B online runs, use the run URL printed by training. For offline runs,
upload the local run files after training:

    wandb sync "${OUT_DIR}/log/wandb"

Runner artifacts should stay under `${training.out_dir}`. The canonical
checkpoint directory is `${training.out_dir}/checkpoints`; older
`${training.out_dir}/ckpt/latest.ckpt` files are still recognized for resume.
Grouped training writes `${training.out_dir}/resolved_config.yaml` and
`${training.out_dir}/run_manifest.json` during runner setup.

Cold-start warmup launchers run a two-stage flow: first collect generated
rollouts, then point `offline_warmup.data_dir` and `offline_warmup.hidden_dir`
at the collected output for WM/classifier warmup. The Ray variant uses
`experiment=collect_rollouts_ray`; the no-Ray variant uses
`experiment=collect_rollouts_onetraj`. Both accept Hydra overrides such as
`task=goal|object|spatial|10` and `run_root=...`. Core runtime controls are
direct Hydra keys on the launcher: `collect.episodes_per_task`,
`collect.episode_horizon`, no-Ray `collect.envs_per_gpu`, Ray
`collect.num_workers`, `warmup.wm_steps`, `warmup.classifier_steps`, and
`warmup.total_env_steps`. The release default keeps
`warmup.total_env_steps=0`; raise it only when you intentionally opt into online
cotrain.

    bash scripts/e2e_coldstart_warmup_cotrain_ray.sh dry_run=true
    bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=spatial dry_run=true

## Evaluation

| Script | Purpose |
| --- | --- |
| `eval/launch_openvla_oft_official_libero_eval.sh` | OpenVLA-OFT eval launcher via `configs/scripts/openvla_oft_official_eval.yaml` |

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamervla.diagnostics.eval_openvla_oft_libero` | OpenVLA-OFT eval implementation |
| `dreamervla.diagnostics.openvla_oft_obs_action_policy` | OpenVLA-OFT policy adapter |
| `dreamervla.diagnostics.eval_frozen_wm_actor` | Frozen-WM actor evaluation |

## Advanced Training

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamervla.runners.online_dreamervla` | Online WMPO experiment loop |
| `dreamervla.runners.frozen_wm_actor_critic` | Frozen-WM actor-critic experiment |
| `dreamervla.runners.collect_online_rollouts_for_classifier` | Collect online rollout shards for classifier experiments |

## Diagnostics And Smoke Tests

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamervla.diagnostics.monitor_dreamervla_metrics` | Summarize training logs |
| `dreamervla.diagnostics.analyze_rynn_hidden_action_metrics` | Hidden/action mismatch analysis |
| `dreamervla.diagnostics.analyze_compact_token_z_reconstruction` | Compact-token reconstruction analysis |
| `dreamervla.diagnostics.compare_action_chunks` | Policy action comparison |
| `dreamervla.diagnostics.compare_policy_trace_runs` | Compare policy trace runs |
| `dreamervla.diagnostics.diagnose_dreamervla_latent_distribution` | DreamerVLA latent distribution diagnostics |
| `dreamervla.diagnostics.diagnose_hidden_token_structure` | Hidden token structure diagnostics |
| `dreamervla.diagnostics.diagnose_ppo_imagine_vs_real` | PPO imagined-vs-real diagnostics |
| `dreamervla.diagnostics.diagnose_residual_cosine` | Residual cosine diagnostics |
| `dreamervla.diagnostics.eval_chunkwm_closeloop` | Chunk-WM closed-loop eval |
| `dreamervla.diagnostics.finetune_reward_head_sparse` | Sparse reward-head finetuning |
| `dreamervla.diagnostics.measure_real_vs_imagine` | Real-vs-imagined rollout comparison |
| `dreamervla.diagnostics.measure_recon_and_action_delta` | Reconstruction and action-delta metrics |
| `dreamervla.diagnostics.measure_reward_and_drift` | Reward and action drift analysis |
| `dreamervla.diagnostics.measure_wm_closed_loop` | WM closed-loop fidelity |
| `dreamervla.diagnostics.measure_wm_imagine_actor` | WM imagined actor diagnostics |
| `dreamervla.diagnostics.measure_wm_imagine_fidelity` | WM imagined-vs-demo fidelity |
| `dreamervla.diagnostics.reward_landscape_sweep` | Reward landscape sweep |
| `dreamervla.diagnostics.validate_oft_rynn_style_sidecar` | Sidecar schema validation |
| `dreamervla.diagnostics.validate_real_rollout_relabel` | Real-rollout relabel validation |
| `dreamervla.diagnostics.visualize_dreamervla_reward` | Reward visualization |
| `dreamervla.diagnostics.smoke_libero_online_env` | LIBERO online env smoke test |

## Legacy Utilities

These modules are kept for reproducibility of older classifier-shard
experiments and are not part of the main release pipeline.

| Module | Purpose |
| --- | --- |
| `dreamervla.legacy.build_classifier_shards_from_demos` | Build old WebDataset classifier shards from demo sidecars |
| `dreamervla.legacy.libero_sim_rollout_shards` | Read old WebDataset classifier shards |

## Conventions

- Use `DVLA_DATA_ROOT` for data location.
- `DVLA_DATA_ROOT` overrides the data root; if unset, scripts use `${DVLA_ROOT}/data`.
- Use `experiment=<name>` for experiment selection.
- Pass Hydra overrides after launcher arguments.
- Keep runtime outputs under `${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/`.
- See `docs/data_layout.md` for the full runtime data layout.
