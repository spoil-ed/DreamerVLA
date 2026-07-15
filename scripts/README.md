# Script registry

Shell files in this directory are entrypoints, not configuration sources. Training and
evaluation defaults live in `configs/`; pass changes as Hydra `key=value` overrides.
Repository data paths are described in `docs/data_layout.md`.

Upload one completed offline W&B run after authenticating once:

```bash
wandb login
bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb
```

For noninteractive authentication, provide `WANDB_API_KEY` instead. No entity or
project argument is required because the uploader reads that metadata from the
offline run.

- `utils/wandb_sync.sh`

Training and evaluation entries under `scripts/experiments/` select recipes directly
from `configs/experiment/`. The only configs under `configs/scripts/` are grouped as
`install/`, `download/`, and `preprocess/` workflows.

## Frozen bootstrap surface

The shell files in `download/`, `install/`, and `preprocess/` are the finalized
bootstrap surface for this repository. These files
must not be modified, renamed, reordered, or copied in part. Run them through the registered entrypoints and change supported
inputs only through Hydra overrides or the environment variables already declared by
the workflow configs.

This restriction applies to every `*.sh` file in those three directories, including
the unnumbered preprocessing wrappers. Keep the wrapper, Hydra config, numbered stage,
tests, and documentation as one indivisible contract. New training or evaluation
behavior belongs in `dreamervla/` and `configs/`, not in these frozen bootstrap files.

## Workflow entrypoints

- `install_env.sh`
- `download_assets.sh`
- `preprocess_libero.sh`

These run the Hydra install, download, and LIBERO preprocessing workflows,
respectively.

The workflow implementation steps are:

- `install/00_apt_tools.sh`
- `install/10_conda_env.sh`
- `install/20_torch.sh`
- `install/30_python_deps.sh`
- `install/40_third_party.sh`
- `install/50_special_packages.sh`
- `install/60_verify.sh`
- `download/00_openvla_oft.sh`
- `download/10_openvla_oft_one_trajectory.sh`
- `download/20_libero_dataset.sh`
- `download/30_calvin_dataset.sh`
- `preprocess/00_hdf5_reward.sh`
- `preprocess/10_oft_hidden_token.sh`
- `preprocess/20_validate.sh`
- `preprocess/prepare_libero_data.sh`
- `preprocess/process_all_libero_data.sh`
- `preprocess/validate_libero_data.sh`

## Experiment entrypoints

Mainline rollout collection:

- `experiments/collect_rollouts/train.sh`

Classifier training and artifact evaluation:

- `experiments/classifier_training/train.sh`
- `experiments/classifier_training/eval.sh`

Official OpenVLA-OFT evaluation:

- `experiments/openvla_oft_official_eval/eval.sh`

Single-trajectory world-model overfit diagnostic:

- `experiments/single_trajectory_overfit/train.sh`
- `experiments/single_trajectory_overfit/eval.sh`

Full-dataset world-model training, profiling, and evaluation:

- `experiments/world_model_training/train.sh`
- `experiments/world_model_training/profile.sh`
- `experiments/world_model_training/eval.sh`

Select the world-model implementation through its Hydra experiment config:

```bash
bash scripts/experiments/world_model_training/train.sh --config dino-wm
bash scripts/experiments/world_model_training/train.sh --config dreamer-wm
```

The model configs live at `configs/worldmodel/dino-wm.yaml` and
`configs/worldmodel/dreamer-wm.yaml`; the same-named files under
`configs/experiment/` are thin experiment selectors. DINO-WM and Dreamer-WM both
use a per-rank batch size of 16 and learning rate `3e-5`; DINO-WM applies that
rate to both its predictor and conditioning optimizers. Training and optimizer
parameters remain under `configs/experiment/`; the shell launcher contains no
training defaults.

Trainable VLA + world-model + classifier cotrain:

- `experiments/cotrain/train.sh`
- `experiments/cotrain/eval.sh`

Cotrain freezes WM/CLS, so both `--wm_ckpt` and `--cls_ckpt` are required unless
the whole run is resumed:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

Checkpoints are written below the owning run root as
`checkpoints/global_step_<N>/manual_cotrain.ckpt`, with
`checkpoints/latest.ckpt` updated for discovery. Resume accepts a run root, a
checkpoint directory, or a checkpoint file and continues in the original run root:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --resume /path/to/openvla_libero/<timestamp>
```

Evaluation requires a policy checkpoint as a Hydra override. Its fixed 100-episode,
25-environment protocol is defined by `configs/experiment/eval_cotrain.yaml`:

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/manual_cotrain.ckpt
```
