# WM Single-Trajectory Overfit Verification

## Goal

Provide one experiment command that proves the configured world model can learn by
overfitting one LIBERO trajectory. The experiment starts the world model from random
initialization, repeatedly trains on every valid sliding window from one HDF5 demo,
and evaluates reconstruction quality with hidden-state MSE and cosine similarity.

This is a model and training-path sanity check. It is not a classifier experiment,
an action-sensitivity study, a general multi-trajectory trainer, or a production
Hydra runner.

## Interface

The user-facing command is:

```bash
CUDA_VISIBLE_DEVICES=7 \
  bash scripts/experiments/wm_single_trajectory_overfit.sh --run
```

The shell script remains a thin launcher for
`python -m dreamervla.diagnostics.wm_single_trajectory_overfit`. Without `--run`,
the command validates and prints the resolved plan without starting GPU training.
The diagnostic composes the repository's static Hydra source with
`experiment=openvla_onetraj_libero_cotrain_noray` and
`task=openvla_onetraj_libero`, then instantiates `cfg.world_model`. It does not depend
on a prior run's `resolved_config.yaml`. CLI overrides select the task, HDF5 filename,
demo key, output directory, maximum epochs, batch size, learning rate, evaluation
interval, and success thresholds. Data paths come from
`task.openvla_oft.input_token_hidden_dir` and `task.hdf5_reward_dir`, which resolve
relative to `DVLA_DATA_ROOT`; no machine-specific absolute data path is embedded in
the implementation.

## Training And Evaluation

The diagnostic constructs the configured world model without loading a checkpoint.
For the selected demo, it reads `obs_embedding`, `lang_emb`, actions, rewards, and
proprio observations. One epoch shuffles and consumes every valid `H+K` sliding
window exactly once.

Every five epochs, the model runs in evaluation mode over every sliding window and
records mean hidden-state MSE and cosine similarity. The default success condition is
both `MSE <= 0.03` and `cosine similarity >= 0.95` for three consecutive evaluations.
Training stops after success or after 200 epochs. Reaching the epoch limit without
meeting the condition is recorded as `not_converged` and returns a non-zero status.

The terminal prints concise epoch progress and every evaluation result so a healthy
run never appears idle.

## Outputs

One output directory contains:

- `metrics.jsonl` with epoch and evaluation records;
- `summary.json` and `summary.md` with final status and best metrics;
- `checkpoints/best.ckpt` and `checkpoints/final.ckpt`;
- `overfit_curves.png` showing training loss, MSE, and cosine similarity.

The best checkpoint is selected by evaluation MSE, with cosine similarity as the
tie-breaker. An exception writes `error.txt` before propagating the failure.

## Tests

Focused CPU tests cover argument planning, exact sliding-window coverage, consecutive
success detection, maximum-epoch failure, and dry-run behavior with temporary HDF5
fixtures. Verification does not launch a long GPU training job.
