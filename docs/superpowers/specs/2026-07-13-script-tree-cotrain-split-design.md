# Script Tree Simplification and Cotrain Split Design

**Date:** 2026-07-13

## Context

DreamerVLA currently exposes many top-level and experiment-specific shell
entrypoints. The supported repository surface should be reduced to installation,
download, preprocessing, and one cotrain workflow with separate training and
evaluation commands.

The existing `e2e_wmcls_cotrain_eval_oneclick.sh` couples training to periodic
real-LIBERO evaluation. Its launcher segments training at evaluation boundaries.
The replacement must make training and evaluation independent jobs.

## Goals

1. Preserve every download, install, and preprocessing shell script unchanged.
2. Replace the coupled cotrain one-click script with independent `train.sh` and
   `eval.sh` entrypoints under `scripts/experiments/cotrain/`.
3. Remove every other shell entrypoint from `scripts/`.
4. Allow cotrain WM/classifier training from either an explicit warm-start pair
   or random initialization.
5. Require evaluation callers to select one explicit cotrain checkpoint.
6. Remove active documentation and test references to deleted entrypoints.

## Final Shell Surface

The protected shell surface is:

- every file under `scripts/download/`;
- every file under `scripts/install/`;
- every file under `scripts/preprocess/`;
- `scripts/download_assets.sh`;
- `scripts/install_env.sh`;
- `scripts/preprocess_libero.sh`.

These files must remain byte-for-byte unchanged.

The only experiment shell entrypoints are:

- `scripts/experiments/cotrain/train.sh`;
- `scripts/experiments/cotrain/eval.sh`.

All other `*.sh` files under `scripts/` are removed. `scripts/README.md` remains
and is rewritten as the registry for the reduced surface.

## Cotrain Training Contract

`scripts/experiments/cotrain/train.sh` is a thin launcher for a dedicated
train-only Hydra recipe named `dreamervla_wmcls_cotrain_ray`. The existing
trainable WM/classifier recipe is retained, but its periodic-evaluation settings
are disabled:

- `manual_cotrain.eval_interval_global_steps=0`;
- `manual_cotrain.eval_initial_global_step=false`.

The old `dreamervla_wmcls_cotrain_ray_eval` recipe is replaced by this
train-only name so the public configuration name matches its behavior.

The training launcher keeps the current staged full-VLA behavior, eight-GPU
default, 20,000-step default, checkpoint cadence, resume support, and normal
Hydra override forwarding. It no longer contains dated `/inspire/...` warm-state
defaults.

WM and classifier initialization is an atomic pair:

- both `WORLD_MODEL_CKPT` and `CLASSIFIER_CKPT` are set: resolve and load both;
- neither variable is set: omit both initialization overrides, so Hydra-built WM
  and classifier modules keep their random initial weights;
- exactly one variable is set: exit with a clear error before launching Ray.

Random initialization applies only to the world model and classifier. The VLA
continues to initialize from the canonical OpenVLA-OFT one-trajectory checkpoint
selected by the task configuration.

The existing frozen-model recipes retain their requirement for explicit frozen
WM/classifier checkpoints. Optional checkpoints apply only to the new trainable
WM/classifier cotrain recipe.

## Cotrain Evaluation Contract

`scripts/experiments/cotrain/eval.sh` runs one independent real-LIBERO evaluation
job. It does not inspect a training directory and never chooses a checkpoint
automatically.

The caller must set:

```bash
COTRAIN_CKPT=/path/to/manual_cotrain.ckpt \
  bash scripts/experiments/cotrain/eval.sh
```

The script fails before Python startup when `COTRAIN_CKPT` is absent or is not a
file. It invokes the existing `eval_libero_vla` launcher with:

- `eval.ckpt_kind=vla_policy`;
- strict component loading;
- the canonical OpenVLA-OFT base checkpoint as `init.vla_ckpt_path`;
- all ten LIBERO Goal tasks;
- ten episodes per task, for 100 total episodes;
- 25 evaluation environments;
- the existing `rlinf_chunk` action, history, reset, and OSMesa protocol;
- cotrain diagnostics enabled.

The output directory is independently configurable and defaults below
`${DVLA_DATA_ROOT}/outputs/eval/cotrain/`. Extra `key=value` arguments are
forwarded after defaults so an explicit caller override wins.

## Error Handling

- A partial WM/classifier warm-start pair is rejected with exit code 2.
- Invalid warm-start paths are rejected by the Python launcher before Ray starts.
- A missing or invalid `COTRAIN_CKPT` is rejected with exit code 2.
- Hydra remains responsible for validating GPU counts, model shapes, and eval
  protocol values.

## Documentation and Compatibility

Active entrypoint registries and tutorials are updated to show only the reduced
shell surface. Historical design and plan records may keep historical command
examples when clearly archival. `spec/99_manual_notes.md` is not modified.

No compatibility wrappers are retained for deleted scripts. Users invoke the new
train/eval paths directly.

## Tests

The implementation is verified with tests that:

1. assert the exact reduced shell tree;
2. assert all protected download/install/preprocess scripts are unchanged;
3. compose the new train-only Hydra recipe with periodic evaluation disabled;
4. prove both warm-start paths are forwarded when both are set;
5. prove neither initialization override is emitted when both are absent;
6. prove a partial warm-start pair fails;
7. prove `eval.sh` rejects a missing checkpoint;
8. prove `eval.sh` builds a strict 100-episode `vla_policy` evaluation command;
9. scan active documentation for references to removed shell entrypoints;
10. run the focused launcher, config, script-hygiene, and runner test suites.

## Non-Goals

- Changing download, installation, or preprocessing behavior.
- Changing the staged cotrain algorithm or model topology.
- Randomly initializing the OpenVLA policy.
- Automatically selecting an evaluation checkpoint.
- Removing Python diagnostics or frozen-model implementation code solely because
  their shell wrappers are removed.
