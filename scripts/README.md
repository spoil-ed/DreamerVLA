# Script registry

Shell files in this directory are entrypoints, not configuration sources. Training and
evaluation defaults live in `configs/`; pass changes as Hydra `key=value` overrides.
Repository data paths are described in `docs/data_layout.md`.

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

- `install_env.sh` runs the Hydra install workflow.
- `download_assets.sh` runs the Hydra download workflow.
- `preprocess_libero.sh` runs the Hydra LIBERO preprocessing workflow.

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

Classifier training and artifact evaluation:

- `experiments/classifier_training/train.sh`
- `experiments/classifier_training/eval.sh`

Single-trajectory world-model overfit diagnostic:

- `experiments/single_trajectory_overfit/train.sh`
- `experiments/single_trajectory_overfit/eval.sh`

Full-dataset world-model training, profiling, and evaluation:

- `experiments/world_model_training/train.sh`
- `experiments/world_model_training/profile.sh`
- `experiments/world_model_training/eval.sh`

The default training entrypoint reproduces DINO-WM dynamics directly over the
persisted OpenVLA-OFT token grid. Its architecture, trajectory slicing
(`frameskip=5` with concatenated actions), normalization, optimizer, and epoch
parameters live in `configs/experiment/wm_dino_token_official.yaml`. The retained
Chunk-WM recipe can still be selected explicitly with
`experiment=wm_official_upper_bound`; its mainline config normalizes raw sidecar
tokens once before transition modeling.

Trainable VLA + world-model + classifier cotrain:

- `experiments/cotrain/train.sh`
- `experiments/cotrain/eval.sh`

Cotrain starts WM/CLS from random weights when both checkpoint environment variables
are absent. Set both `WORLD_MODEL_CKPT` and `CLASSIFIER_CKPT` to warm-start them:

```bash
bash scripts/experiments/cotrain/train.sh

WORLD_MODEL_CKPT=/path/to/wm.ckpt \
CLASSIFIER_CKPT=/path/to/classifier.ckpt \
  bash scripts/experiments/cotrain/train.sh
```

Evaluation requires a policy checkpoint as a Hydra override. Its fixed 100-episode,
25-environment protocol is defined by `configs/experiment/eval_cotrain.yaml`:

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/manual_cotrain.ckpt
```
