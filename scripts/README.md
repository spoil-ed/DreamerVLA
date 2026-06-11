# Script Registry

`scripts/` contains resumable shell launchers only. Python implementation code
lives under the `dreamer_vla` package and is launched with `python -m`.

## Main Path

| Script | Purpose |
| --- | --- |
| `install_env.sh` | Run resumable install steps under `scripts/install/` |
| `download_assets.sh` | Download checkpoints and LIBERO / CALVIN data |
| `preprocess_libero.sh` | Compatibility wrapper that preprocesses the standard LIBERO suites |
| `preprocess/prepare_libero_data.sh` | Build LIBERO HDF5 views, reward labels, manifests, and action-hidden sidecars |
| `train_vla.sh` | VLA SFT via Hydra route configs |
| `train_wm.sh` | World-model and classifier training via Hydra route configs |
| `train_dreamervla.sh` | DreamerVLA training via Hydra route configs |
| `eval_libero_vla.sh` | LIBERO rollout eval for VLA or Dreamer checkpoints |

## Install Steps

| Script | Purpose |
| --- | --- |
| `install/00_apt_tools.sh` | System packages |
| `install/10_conda_env.sh` | Conda environment |
| `install/20_python_deps.sh` | PyTorch, repo package, Python deps, flash-attn |
| `install/30_third_party.sh` | LIBERO / robosuite stack |
| `install/40_verify.sh` | Import and CUDA visibility check |
| `install/_env.sh` | Shared install-step environment |

## Preprocessing

| Script | Purpose |
| --- | --- |
| `preprocess_libero.sh` | Top-level wrapper around `preprocess/prepare_libero_data.sh` for one or more LIBERO suites |
| `preprocess/prepare_libero_data.sh` | End-to-end resumable LIBERO preprocessing path |
| `preprocess/process_all_libero_data.sh` | Lower-level LIBERO image, conversation, token, and config generation |
| `preprocess/concat_record_libero.sh` | Concatenate LIBERO record files |

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.preprocess.filter_marked_libero_hdf5` | Filter no-op marked HDF5 files |
| `dreamer_vla.preprocess.preprocess_remaining_steps_reward` | Remaining-steps reward labels |
| `dreamer_vla.preprocess.preprocess_rynn_pixel_hidden` | RynnVLA action-hidden sidecar extraction |
| `dreamer_vla.preprocess.preprocess_oft_action_hidden` | OpenVLA-OFT action-hidden sidecar extraction |

## Evaluation

| Script | Purpose |
| --- | --- |
| `eval/launch_openvla_oft_official_libero_eval.sh` | OpenVLA-OFT eval launcher |

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.evaluation.eval_openvla_oft_libero` | OpenVLA-OFT eval implementation |
| `dreamer_vla.evaluation.openvla_oft_obs_action_policy` | OpenVLA-OFT policy adapter |
| `dreamer_vla.evaluation.eval_frozen_wm_actor` | Frozen-WM actor evaluation |

## Advanced Training

Python modules:

| Module | Purpose |
| --- | --- |
| `dreamer_vla.training.train_online_rynnvla_action_hidden_dreamervla` | Online WMPO experiment loop |
| `dreamer_vla.training.train_online_rynnvla_action_hidden_dreamervla_multiproc` | Multi-process online collector variant |
| `dreamer_vla.training.train_frozen_wm_actor_critic` | Frozen-WM actor-critic experiment |
| `dreamer_vla.training.collect_online_rollouts_for_classifier` | Collect online rollout shards for classifier experiments |

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
| `dreamer_vla.smoke.smoke_libero_online_env` | LIBERO online env smoke test |

## Legacy Utilities

These modules are kept for reproducibility of older classifier-shard
experiments and are not part of the main release pipeline.

| Module | Purpose |
| --- | --- |
| `dreamer_vla.legacy.build_classifier_shards_from_demos` | Build old WebDataset classifier shards from demo sidecars |
| `dreamer_vla.legacy.libero_sim_rollout_shards` | Read old WebDataset classifier shards |

## Conventions

- Use `DVLA_DATA_ROOT` for data location.
- Use `CONFIG=<route>` for route selection.
- Pass Hydra overrides after launcher arguments.
- Keep runtime outputs under `${DVLA_DATA_ROOT:-data}/outputs/`.
- See `docs/data_layout.md` for the full runtime data layout.
