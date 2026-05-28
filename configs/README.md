# Config Registry

Hydra configs are now grouped by training route. Shell launchers should choose
one concrete config and keep operational overrides small: GPUs, batch size,
checkpoint paths, output tags, and smoke-test limits.

## Formal Entrypoints

| Stage | Script | Default Config |
| --- | --- | --- |
| VLA training | `scripts/train_vla.sh` | `vla_pi0_query` |
| VLA one-trajectory SFT | `CONFIG=vla_sft_one_trajectory scripts/train_vla.sh` | `vla_sft_one_trajectory` |
| OpenVLA-OFT one-trajectory SFT | `CONFIG=openvla_oft_hdf5_one_trajectory scripts/train_vla.sh` | `openvla_oft_hdf5_one_trajectory` |
| WM training | `scripts/train_wm.sh` | `world_model_dinowm_chunk` |
| DreamerVLA training | `scripts/train_dreamervla.sh` | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| LIBERO eval | `scripts/eval_libero_vla.sh` | `eval_libero_vla` |

## Formal Configs

| Route | Config |
| --- | --- |
| VLA | `vla_pi0_query` |
| VLA one-trajectory SFT | `vla_sft_one_trajectory` |
| OpenVLA-OFT HDF5 SFT | `openvla_oft_hdf5` |
| OpenVLA-OFT LM-head one-trajectory SFT | `openvla_oft_hdf5_one_trajectory` |
| WM DINO step | `world_model_dinowm_step` |
| WM DINO chunk | `world_model_dinowm_chunk` |
| DreamerVLA PPO, DINO-WM step | `dreamervla_rynn_dino_wm_actor_critic` |
| DreamerVLA PPO, DINO-WM chunk/outcome | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| DreamerVLA PPO, OpenVLA-OFT chunk/outcome | `dreamervla_oft_dino_wm_wmpo_outcome` |
| Online DreamerVLA WMPO outcome | `online_wmpo_outcome_libero_goal` |
| LIBERO rollout eval | `eval_libero_vla` |

Formal configs use Hydra defaults to include the task config:

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

## Archive

Unconfirmed or historical configs live under:

```text
archive/uncertain_configs/
archive/libero10_legacy/
```

Do not use archived configs as defaults without checking paths, checkpoint
compatibility, and sidecar schema.
