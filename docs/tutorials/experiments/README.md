# Experiment tutorials

Step-only end-to-end recipes (clean checkout → install → download → preprocess →
train → eval). **All background, rationale, memory/OOM, WM sizing and logging notes
are in [EXPLAINED.md](EXPLAINED.md).** Parameter reference:
[PARAMETERS.md](../../PARAMETERS.md).

Normal changes are Hydra overrides, e.g.
`gpus=0,1 ngpu=2 batch_size=16 num_workers=4 num_epochs=20 out_dir=/tmp/run`.

## Recipes

| Pipeline | Hydra `task=` | Main configs |
| --- | --- | --- |
| [RynnVLA_LIBERO](RynnVLA_LIBERO.md) | `rynnvla_libero` | `world_model_chunk`, `dreamervla_rynn_wm_lumos` |
| [OpenVLA one-traj cold-start cotrain](OpenVLA_Onetraj_LIBERO.md) | `goal\|object\|spatial\|10` launcher shorthand | `collect_rollouts_ray`, `oft_discrete_token_world_model_chunk`, `dreamervla_oft_discrete_token_wm_lumos`, `openvla_onetraj_libero_cotrain_ray`, `eval_libero_vla` |
| [Ray/manual cotrain backend](../../../spec/04_complete_loop.md) | synthetic / gated real smoke | `manual_cotrain_ray_*`, legacy `online_cotrain_ray_*` |
| [WM single-episode overfit probe](wm_single_episode_overfit.py) | `libero_goal` HDF5 + hidden sidecar | diagnostic script; dry-run unless `--run` is passed |

The `task=` token is snake_case; on-disk data artifacts keep their historical
`task.artifact_name` directories (e.g. `OpenVLA_Onetraj_LIBERO_libero_goal`), so paths
inside the commands mix the two — this is intentional (see EXPLAINED.md).

## Diagnostics

The WM single-episode overfit probe is intentionally not a launcher and does not run
unless `--run` is passed:

```bash
python docs/tutorials/experiments/wm_single_episode_overfit.py
```

To run it on an explicitly selected GPU:

```bash
CUDA_VISIBLE_DEVICES=7 \
  python docs/tutorials/experiments/wm_single_episode_overfit.py --run \
  --out-dir "${DVLA_DATA_ROOT}/outputs/world_model_probe/single_episode_overfit"
```

It records `metrics.jsonl` with `true`, `zero`, and `random` action-chunk rollout
comparisons so action sensitivity can be checked without starting cotrain.
