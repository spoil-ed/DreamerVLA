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
| [OpenVLA one-traj cold-start cotrain](OpenVLA_Onetraj_LIBERO.md) | `goal\|object\|spatial\|10` launcher shorthand | `collect_rollouts`, `openvla_onetraj_libero_cotrain`, `wm_full_dataset_train`, `eval_libero_vla` |
| [Ray/manual cotrain backend](../../../spec/04_complete_loop.md) | synthetic / gated real smoke | `manual_cotrain_ray_*`, `openvla_onetraj_libero_cotrain` |
| [WM single-trajectory overfit probe](../../../scripts/experiments/single_trajectory_overfit/train.sh) | `libero_goal` HDF5 + hidden sidecar | diagnostic script with a check before training |

The `task=` token is snake_case; on-disk data artifacts keep their
`task.artifact_name` directories (e.g. `OpenVLA_Onetraj_LIBERO_libero_goal`), so paths
inside the commands mix the two — this is intentional (see EXPLAINED.md).

## Diagnostics

The WM single-trajectory overfit probe checks inputs first, then runs training:

```bash
bash scripts/experiments/single_trajectory_overfit/train.sh
```

To run it on an explicitly selected GPU:

```bash
CUDA_VISIBLE_DEVICES=7 \
  bash scripts/experiments/single_trajectory_overfit/train.sh \
  --out-dir "${DVLA_DATA_ROOT}/outputs/world_model_probe/single_trajectory_overfit"
```

It records `metrics.jsonl` with `true`, `zero`, and `random` action-chunk rollout
comparisons, then writes `summary.json` and `summary.md` so the final hidden-MSE
trend and action-sensitivity table are visible without opening TensorBoard.
