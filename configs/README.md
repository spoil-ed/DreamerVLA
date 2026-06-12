# Config Registry

DreamerVLA uses the standard Hydra template pattern:

```text
configs/
├── train.yaml
├── experiment/
├── VLA/
├── worldmodel/
├── classifier/
├── dreamervla/
├── evaluation/
├── task/
└── logger/
```

`train.yaml` is the stable entrypoint. `experiment=<name>` selects a recipe
under `configs/experiment/`; that recipe overrides one cohesive module group
(`VLA`, `worldmodel`, `classifier`, `dreamervla`, or `evaluation`). Keep shell
overrides operational and small: GPUs, batch size, checkpoint paths, output
tags, and smoke-test limits.

`logger=tensorboard` is the default for grouped training and writes local
TensorBoard event files under `${training.out_dir}/log/tensorboard`.
Use `logger=wandb` to send main-process metrics to W&B online mode while
keeping W&B run files under `${training.out_dir}/log/wandb`:

```bash
python -m dreamer_vla.train experiment=world_model_dinowm_chunk logger=wandb
```

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| VLA training | `scripts/train_vla.sh` | `vla_rynnvla_action_head` |
| VLA one-trajectory SFT | `scripts/train_vla.sh experiment=vla_sft_one_trajectory` | `vla_sft_one_trajectory` |
| OpenVLA-OFT one-trajectory SFT | `scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory` | `openvla_oft_hdf5_one_trajectory` |
| OpenVLA-OFT L1 one-trajectory SFT | `scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory_l1` | `openvla_oft_hdf5_one_trajectory_l1` |
| WM training | `scripts/train_wm.sh` | `world_model_dinowm_chunk` |
| DreamerVLA training | `scripts/train_dreamervla.sh` | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| LIBERO eval | `scripts/eval_libero_vla.sh` | `eval_libero_vla` |

## Experiments

| Experiment | Module group |
| --- | --- |
| `vla_rynnvla_action_head` | `VLA/rynnvla_action_head` |
| `vla_sft_one_trajectory` | `VLA/rynnvla_one_trajectory` |
| `openvla_oft_hdf5` | `VLA/openvla_oft` |
| `openvla_oft_hdf5_one_trajectory` | `VLA/openvla_oft_one_trajectory` |
| `openvla_oft_hdf5_one_trajectory_l1` | `VLA/openvla_oft_l1_one_trajectory` |
| `world_model_dinowm_step` | `worldmodel/rynnvla_action_step` |
| `world_model_dinowm_chunk` | `worldmodel/rynnvla_action_chunk` |
| `world_model_dinowm_chunk_input_tokens` | `worldmodel/rynnvla_input_token_chunk` |
| `oft_world_model_dinowm_chunk` | `worldmodel/openvla_oft_action_chunk` |
| `oft_world_model_dinowm_chunk_input_tokens` | `worldmodel/openvla_oft_input_token_chunk` |
| `latent_classifier_libero_goal_chunk` | `classifier/rynnvla_action_chunk` |
| `latent_classifier_libero_goal_chunk_input_tokens` | `classifier/rynnvla_input_token_chunk` |
| `oft_latent_classifier_chunk` | `classifier/openvla_oft_action_chunk` |
| `oft_latent_classifier_chunk_input_tokens` | `classifier/openvla_oft_input_token_chunk` |
| `dreamervla_rynn_dino_wm_actor_critic` | `dreamervla/rynnvla_actor_critic` |
| `dreamervla_rynn_dino_wm_wmpo_outcome` | `dreamervla/rynnvla_wmpo_outcome` |
| `dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens` | `dreamervla/rynnvla_input_token_wmpo_outcome` |
| `dreamervla_oft_dino_wm_wmpo_outcome` | `dreamervla/openvla_oft_wmpo_outcome` |
| `dreamervla_oft_dino_wm_wmpo_outcome_input_tokens` | `dreamervla/openvla_oft_input_token_wmpo_outcome` |
| `online_wmpo_outcome_libero_goal` | `dreamervla/online_wmpo_outcome_libero_goal` |
| `eval_libero_vla` | `evaluation/libero_vla` |

Module configs use Hydra defaults to include the task config:

```yaml
defaults:
  - _self_
  - /task: libero_goal
```

Keep concrete dataset task paths, horizons, sidecar expectations, and
task-specific dimensions in `task/*.yaml`. The module configs define the
runner, model, optimizer, and algorithm for that experiment family.

## Task Configs

The task folder contains concrete dataset task definitions only:

```text
task/libero_goal.yaml
task/libero_object.yaml
task/libero_spatial.yaml
task/libero_10.yaml
```

Switch tasks with Hydra, for example:

```bash
bash scripts/train_vla.sh task=libero_object
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_spatial
```
