# Experiment Tutorials

These tutorials are short end-to-end recipes. They start from a clean checkout,
then go through install, download, preprocessing, training, and final LIBERO
eval.

Use Hydra overrides for normal changes:

```bash
gpus=0,1 ngpu=2 batch_size=16 num_workers=4 max_steps=1000 out_dir=/tmp/run
```

The shell launchers are intentionally thin. They set project/data roots and
call the Hydra launcher; all experiment choice is in `experiment=...` and
normal Hydra keys.

## Recipes

| Pipeline | Processed-data task name | Main configs |
| --- | --- | --- |
| [RynnVLA_LIBERO](RynnVLA_LIBERO.md) | `RynnVLA_LIBERO` | `world_model_dinowm_chunk`, `dreamervla_rynn_dino_wm_wmpo_outcome` |
| [OpenVLA_Onetraj_LIBERO](OpenVLA_Onetraj_LIBERO.md) | `OpenVLA_Onetraj_LIBERO` | `oft_world_model_dinowm_chunk`, `dreamervla_oft_dino_wm_wmpo_outcome` |

The pipeline task name is both the Hydra `task=` value and the prefix used for
processed-data intermediate folders and sidecars. The raw benchmark suite is
still `libero_goal` in these tutorials.

Scheme A is the action-slot hidden-token route. Scheme B is the optional
frame-level visual input-token route; the WM receives actions through its
action input and DreamerVLA uses a bridge actor to produce action slots.

Classifier and WMPO training need both success and failure rollout corpora. The
standard LIBERO download gives success demos. If you do not have failure demos
and matching sidecars yet, stop after WM training or use the actor-critic route
where applicable.
