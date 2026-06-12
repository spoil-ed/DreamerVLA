# Config Registry

Hydra configs are now grouped by training route. Shell launchers should choose
one concrete config and keep operational overrides small: GPUs, batch size,
checkpoint paths, output tags, and smoke-test limits.

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| VLA training | `scripts/train_vla.sh` | `vla_rynnvla_action_head` |
| VLA one-trajectory SFT | `CONFIG=vla_sft_one_trajectory scripts/train_vla.sh` | `vla_sft_one_trajectory` |
| OpenVLA-OFT one-trajectory SFT | `CONFIG=openvla_oft_hdf5_one_trajectory scripts/train_vla.sh` | `openvla_oft_hdf5_one_trajectory` |
| OpenVLA-OFT L1 one-trajectory SFT | `CONFIG=openvla_oft_hdf5_one_trajectory_l1 scripts/train_vla.sh` | `openvla_oft_hdf5_one_trajectory_l1` |
| WM training | `scripts/train_wm.sh` | `world_model_dinowm_chunk` |
| DreamerVLA training | `scripts/train_dreamervla.sh` | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| LIBERO eval | `scripts/eval_libero_vla.sh` | `eval_libero_vla` |

## Route Configs

| Route | Config |
| --- | --- |
| VLA | `vla_rynnvla_action_head` |
| VLA one-trajectory SFT | `vla_sft_one_trajectory` |
| OpenVLA-OFT HDF5 SFT | `openvla_oft_hdf5` |
| OpenVLA-OFT LM-head one-trajectory SFT | `openvla_oft_hdf5_one_trajectory` |
| OpenVLA-OFT L1-regression one-trajectory SFT | `openvla_oft_hdf5_one_trajectory_l1` |
| WM DINO step | `world_model_dinowm_step` |
| WM DINO chunk, action-query Scheme A | `world_model_dinowm_chunk` |
| WM DINO chunk, input-token Scheme B | `world_model_dinowm_chunk_input_tokens` |
| DreamerVLA PPO, DINO-WM step | `dreamervla_rynn_dino_wm_actor_critic` |
| DreamerVLA PPO, DINO-WM chunk/outcome Scheme A | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| DreamerVLA PPO, DINO-WM chunk/outcome Scheme B | `dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens` |
| DreamerVLA online WMPO outcome | `online_wmpo_outcome_libero_goal` |
| DreamerVLA PPO, OpenVLA-OFT chunk/outcome Scheme A | `dreamervla_oft_dino_wm_wmpo_outcome` |
| DreamerVLA PPO, OpenVLA-OFT chunk/outcome Scheme B | `dreamervla_oft_dino_wm_wmpo_outcome_input_tokens` |
| RynnVLA latent classifier Scheme A | `latent_classifier_libero_goal_chunk` |
| RynnVLA latent classifier Scheme B | `latent_classifier_libero_goal_chunk_input_tokens` |
| OpenVLA-OFT latent classifier Scheme A | `oft_latent_classifier_chunk` |
| OpenVLA-OFT latent classifier Scheme B | `oft_latent_classifier_chunk_input_tokens` |
| LIBERO rollout eval | `eval_libero_vla` |

Route configs use Hydra defaults to include the task config:

```yaml
defaults:
  - _self_
  - /task: libero_goal
```

Keep concrete dataset task paths, horizons, sidecar expectations, and
task-specific dimensions in `task/*.yaml`. The training configs define the
runner, model, optimizer, and algorithm route.

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
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal
CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_spatial
```
