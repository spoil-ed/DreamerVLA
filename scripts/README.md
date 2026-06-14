# Script Registry

`scripts/` contains resumable shell launchers only. Python implementation code
lives under the `dreamer_vla` package and is launched with `python -m`.

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
| `eval_libero_vla.sh` | LIBERO rollout eval for VLA or Dreamer checkpoints |

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
| `dreamer_vla.launchers.workflow` | Hydra workflow runner for install/download/preprocess shell steps |
| `dreamer_vla.preprocess.filter_marked_libero_hdf5` | Filter no-op marked HDF5 files |
| `dreamer_vla.preprocess.preprocess_remaining_steps_reward` | Remaining-steps reward labels |
| `dreamer_vla.preprocess.validate_libero_data_prep` | Structural validation for HDF5, conv, token, record, manifest, and config counts |
| `dreamer_vla.preprocess.preprocess_rynn_pixel_hidden` | RynnVLA action-hidden sidecar extraction |
| `dreamer_vla.preprocess.preprocess_oft_action_hidden` | OpenVLA-OFT action-hidden sidecar extraction |

Common launcher flags stay intentionally small:

    bash scripts/install_env.sh only=[20_torch] force=true
    bash scripts/download_assets.sh download.rynnvla=false download.libero=true env.LIBERO_SUITES=libero_goal
    bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1 num_procs=8

    bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
      gpus=0,1 ngpu=2 batch_size=16 num_workers=4 training.max_steps=1000

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
Grouped training defaults to `logger=tensorboard` and writes local TensorBoard
events under `${training.out_dir}/log/tensorboard`; use `logger=wandb` to route
main-process metrics through W&B online mode. Use `logger=tensorboard_wandb`
when you want local TensorBoard events and online W&B tracking for the same
run.

Runner artifacts should stay under `${training.out_dir}`. The canonical
checkpoint directory is `${training.out_dir}/checkpoints`; older
`${training.out_dir}/ckpt/latest.ckpt` files are still recognized for resume.
Grouped training writes `${training.out_dir}/resolved_config.yaml` and
`${training.out_dir}/run_manifest.json` during runner setup.

## Evaluation

| Script | Purpose |
| --- | --- |
| `eval/launch_openvla_oft_official_libero_eval.sh` | OpenVLA-OFT eval launcher |

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.diagnostics.eval_openvla_oft_libero` | OpenVLA-OFT eval implementation |
| `dreamer_vla.diagnostics.openvla_oft_obs_action_policy` | OpenVLA-OFT policy adapter |
| `dreamer_vla.diagnostics.eval_frozen_wm_actor` | Frozen-WM actor evaluation |

## Advanced Training

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.runners.online_dreamervla` | Online WMPO experiment loop |
| `dreamer_vla.runners.online_dreamervla_multiproc` | Multi-process online collector variant |
| `dreamer_vla.runners.frozen_wm_actor_critic` | Frozen-WM actor-critic experiment |
| `dreamer_vla.runners.collect_online_rollouts_for_classifier` | Collect online rollout shards for classifier experiments |

## Diagnostics And Smoke Tests

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.diagnostics.monitor_dreamer_vla_metrics` | Summarize training logs |
| `dreamer_vla.diagnostics.analyze_rynn_hidden_action_metrics` | Hidden/action mismatch analysis |
| `dreamer_vla.diagnostics.analyze_compact_token_z_reconstruction` | Compact-token reconstruction analysis |
| `dreamer_vla.diagnostics.compare_action_chunks` | Policy action comparison |
| `dreamer_vla.diagnostics.compare_policy_trace_runs` | Compare policy trace runs |
| `dreamer_vla.diagnostics.diagnose_dreamervla_latent_distribution` | DreamerVLA latent distribution diagnostics |
| `dreamer_vla.diagnostics.diagnose_hidden_token_structure` | Hidden token structure diagnostics |
| `dreamer_vla.diagnostics.diagnose_ppo_imagine_vs_real` | PPO imagined-vs-real diagnostics |
| `dreamer_vla.diagnostics.diagnose_residual_cosine` | Residual cosine diagnostics |
| `dreamer_vla.diagnostics.eval_chunkwm_closeloop` | Chunk-WM closed-loop eval |
| `dreamer_vla.diagnostics.finetune_reward_head_sparse` | Sparse reward-head finetuning |
| `dreamer_vla.diagnostics.measure_real_vs_imagine` | Real-vs-imagined rollout comparison |
| `dreamer_vla.diagnostics.measure_recon_and_action_delta` | Reconstruction and action-delta metrics |
| `dreamer_vla.diagnostics.measure_reward_and_drift` | Reward and action drift analysis |
| `dreamer_vla.diagnostics.measure_wm_closed_loop` | WM closed-loop fidelity |
| `dreamer_vla.diagnostics.measure_wm_imagine_actor` | WM imagined actor diagnostics |
| `dreamer_vla.diagnostics.measure_wm_imagine_fidelity` | WM imagined-vs-demo fidelity |
| `dreamer_vla.diagnostics.reward_landscape_sweep` | Reward landscape sweep |
| `dreamer_vla.diagnostics.validate_oft_rynn_style_sidecar` | Sidecar schema validation |
| `dreamer_vla.diagnostics.validate_real_rollout_relabel` | Real-rollout relabel validation |
| `dreamer_vla.diagnostics.visualize_dreamervla_reward` | Reward visualization |
| `dreamer_vla.diagnostics.smoke_libero_online_env` | LIBERO online env smoke test |

## Legacy Utilities

These modules are kept for reproducibility of older classifier-shard
experiments and are not part of the main release pipeline.

| Module | Purpose |
| --- | --- |
| `dreamer_vla.legacy.build_classifier_shards_from_demos` | Build old WebDataset classifier shards from demo sidecars |
| `dreamer_vla.legacy.libero_sim_rollout_shards` | Read old WebDataset classifier shards |

## Conventions

- Use `DVLA_DATA_ROOT` for data location.
- `DVLA_DATA_ROOT` is independent of `DVLA_ROOT`; if unset, scripts use relative `data`.
- Use `experiment=<name>` for experiment selection.
- Pass Hydra overrides after launcher arguments.
- Keep runtime outputs under `${DVLA_DATA_ROOT:-data}/outputs/`.
- See `docs/data_layout.md` for the full runtime data layout.
