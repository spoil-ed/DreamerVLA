# Script Entry Points

Scripts are thin wrappers around `python -m dreamer_vla.cli.train` or small diagnostic
tools. Keep durable logic in `dreamer_vla/`; keep scripts as reproducible launch
recipes.

## Data Preparation

Historical data-preparation shell recipes are archived under
`scripts/archive/uncertain_shells/`. Current training launchers take task
metadata from `configs/task/*.yaml`.

## Training

| Script | Main Config | Public Runner | Purpose |
| --- | --- | --- | --- |
| `train_vla.sh` | `vla_pi0_query`, `vla_sft_one_trajectory`, `openvla_oft_hdf5`, `openvla_oft_hdf5_one_trajectory` | route-specific `dreamer_vla.runners.*` target from config | VLA SFT, including one trajectory per task |
| `train_vla_nongoal_45.sh` | `vla_pi0_query` | `VLASFTRunner` | LIBERO non-goal VLA SFT on GPUs 4,5; switch task with `TAG=<tag>` |
| `train_wm.sh` | `world_model_dinowm_chunk`, `world_model_dinowm_step`, `oft_world_model_dinowm_chunk` | route-specific `dreamer_vla.runners.*` target from config | WM training |
| `train_dreamervla.sh` | `dreamervla_rynn_dino_wm_wmpo_outcome`, `dreamervla_rynn_dino_wm_actor_critic`, `dreamervla_oft_dino_wm_wmpo_outcome` | `JointDreamerVLARunner` | DreamerVLA training |

Configs point directly at the route-specific runner class.

Most wrappers accept standard environment variables:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
BATCH_SIZE=96
NUM_WORKERS=2
RUN_TAG=my_run
OUT_DIR_BASE=/path/to/output/root
```

`DETACH=1` backgrounds training and writes a `train.pid`; omit `DETACH` to keep
logs in the terminal.

Non-goal LIBERO VLA SFT uses suite-specific pretrained weights under
`data/ckpts/VLA_model_256/<suite>`:

```bash
bash scripts/train_vla_nongoal_45.sh libero_10
TAG=libero_object bash scripts/train_vla_nongoal_45.sh
TAG=libero_spatial bash scripts/train_vla_nongoal_45.sh
```

One-trajectory SFT keeps one demonstration trajectory per LIBERO task:

```bash
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_object dataset.trajectory_offset=2
CONFIG=openvla_oft_hdf5_one_trajectory bash scripts/train_vla.sh task=libero_10
```

`openvla_oft_hdf5_one_trajectory` uses the OpenVLA-OFT LLaMA LM head for action-token SFT (`policy.use_l1_regression=false`) and samples one demo per task file with `dataset.demo_selection_seed`.

## Evaluation

| Script | Purpose |
| --- | --- |
| `eval_libero_vla.sh` | Single-process LIBERO rollout eval for VLA or Dreamer checkpoints |

For Dreamer checkpoints, `eval_libero_vla.sh` supports:

```text
eval.ckpt_kind=dreamer
eval.dreamer_policy_source=ckpt|init
eval.dreamer_actor_input_source=rssm|encoder|encoder_sequence
```

These switches are useful for actor and hidden ablations.

## Diagnostics

Core diagnostics (`scripts/diagnostics/`):

| Script | Purpose |
| --- | --- |
| `analyze_rynn_hidden_action_metrics.py` | Offline hidden/action mismatch metrics |
| `monitor_dreamer_vla_metrics.py` | Summarize training log trends |
| `visualize_dreamervla_reward.py` | Reward-model visualization helper |
| `smoke_libero_online_env.py` | Smoke test for online LIBERO env wiring (lives in `scripts/smoke/`) |

Hidden-token structure (paper §5.1 analysis):

| Script | Purpose |
| --- | --- |
| `diagnose_hidden_token_structure.py` | Phase 1 statistics on 35×1024 action_hidden over time / joint axes |
| `diagnose_residual_cosine.py` | Cross-token residual cosine; tests per-sample redundancy vs LayerNorm bias |
| `analyze_compact_token_z_reconstruction.py` | Reconstruction quality of `CompactTokenSequenceAutoencoder` |

Classifier ceiling estimation:

| Script | Purpose |
| --- | --- |
| `estimate_classifier_ceiling.py` | LR / kNN / small-MLP triangulation of LatentSuccessClassifier Bayes ceiling |
| `finetune_reward_head_sparse.py` | Fine-tune only WM reward head as terminal-success classifier (WMPO recipe) |

WM imagine fidelity:

| Script | Purpose |
| --- | --- |
| `measure_wm_imagine_fidelity.py` | Faithfulness of WM-imagined trajectory under demo actions (feature / reward) |
| `measure_wm_imagine_actor.py` | Same, but under trained / SFT-init / demo actions (OOD test) |
| `measure_recon_and_action_delta.py` | hidden_decoder reconstruction quality + SFT actor sensitivity to recon |
| `measure_reward_and_drift.py` | Reward curve on success demo + SFT-direction reward peak + action drift |

Policy comparison:

| Script | Purpose |
| --- | --- |
| `compare_action_chunks.py` | Trained policy vs frozen pi0-SFT baseline action_chunks on shared WM features |
| `compare_policy_trace_runs.py` | Compare `policy_trace.jsonl` from VLA and DreamerVLA rollouts |
| `diagnose_dreamervla_latent_distribution.py` | DreamerVLA latent distribution diagnostics |

Data validation:

| Script | Purpose |
| --- | --- |
| `validate_oft_rynn_style_sidecar.py` | Validate OFT / rynn-style sidecar HDF5 schema |

Diagnostic outputs should go under:

```text
data/outputs/eval/
```

## Frozen-WM actor / critic route

Frozen-WM is a non-mainline ablation route where the world model is held
constant and only the actor / critic are trained.

| Script | Purpose |
| --- | --- |
| `scripts/training/train_frozen_wm_actor_critic.py` | Train actor / critic with the WM frozen (reuses `OnlineReplay` from the live training script) |
| `scripts/eval/eval_frozen_wm_actor.py` | Deterministic LIBERO rollout eval for checkpoints saved by the above; emits MP4s + JSON summary |

## Preprocess

| Script | Purpose |
| --- | --- |
| `scripts/preprocess/preprocess_oft_action_hidden.py` | Build OFT action-hidden sidecar from raw demos |
| `scripts/preprocess/preprocess_remaining_steps_reward.py` | Precompute `remaining_steps`-style dense reward labels |
| `scripts/preprocess/preprocess_rynn_pixel_hidden.py` | Build rynn pixel / hidden sidecar |
| `scripts/preprocess/build_classifier_shards_from_demos.py` | Pack demo action-hiddens into WebDataset shards for the LatentSuccessClassifier (positives only; failure-class negatives must be appended separately) |
| `scripts/preprocess/collect_online_rollouts_for_classifier.py` | Roll out pi0 SFT in LIBERO sim to collect failure-class negatives for the classifier |

## Script Hygiene

- Do not hard-code a new experiment path if an env var or Hydra override is enough.
- Put long-lived launch defaults in the wrapper; put experiment identity in `RUN_TAG`.
- Keep logs under `data/outputs/...`; do not write logs into the repo root.
- If a script becomes a stable entry point, list it here.
