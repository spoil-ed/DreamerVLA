# Unified Train Launcher Design

## Goal

Expose one public training launcher, `python -m dreamervla.launchers.train`, for
generic experiments and cotrain. Remove `dreamervla.launchers.cotrain` without
moving cotrain-specific checkpoint or topology rules into hard-coded experiment
branches in the generic launcher.

The retained cotrain shell command is:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

## Options Considered

### Concatenate `cotrain.py` into `train.py`

This removes one file but leaves two control flows in one large module and forces
all launch modes to import PyTorch checkpoint inspection code. It also encourages
`if experiment == "openvla_libero"` branches. Reject this option.

### Make every cotrain rule generic YAML data

Simple mappings such as public option names and paired values fit YAML, but
classifier checkpoint inspection and frozen-component validation are executable
contracts. Encoding those algorithms as generic YAML would make the schema
complex and weakly typed. Reject a YAML-only implementation.

### One launcher plus a Hydra-selected contract

Retain generic parsing, resume handling, Hydra composition, environment setup,
command creation, and subprocess execution in `launchers.train`. Select an
optional launch contract through Hydra for specialized validation and override
derivation. This is the chosen design.

## Architecture

`dreamervla.launchers.train` owns a reusable `ExperimentLaunch` value and a
`build_launch(argv)` function. `main()` only prints the resolved launch, honors
dry-run behavior, and executes it. Generic experiments use a no-op contract.

The launcher performs these phases:

1. Read the experiment selection without rejecting contract-specific options.
2. Compose the experiment once to discover `launch.contract._target_`.
3. Instantiate the contract selected by Hydra.
4. Let the contract normalize its public options and environment aliases into
   Hydra overrides.
5. Run the existing generic argument parser, resume mapping, and launcher aliases.
6. Compose the complete resolved configuration.
7. Let the contract derive additional overrides, then recompose if necessary.
8. Validate required values and the selected contract.
9. Build the generic process environment and let the contract enforce specialized
   topology constraints.
10. Build and optionally execute `python -m dreamervla.train`.

No generic launcher branch names `openvla_libero`, `CotrainRunner`, WM, or the
classifier. Selecting the contract is entirely configuration-owned.

## Contract Boundary

Define a narrow launch-contract protocol under `dreamervla/launchers/` with four
operations:

- normalize contract-specific CLI and environment inputs into Hydra overrides;
- derive overrides after initial composition;
- validate the resolved configuration;
- adjust/validate the child-process environment and provide optional summary
  lines.

The cotrain contract owns the behavior currently in `cotrain.py`:

- `--wm_ckpt` and `--cls_ckpt` parsing and existing-path validation;
- `WORLD_MODEL_CKPT` and `CLASSIFIER_CKPT` compatibility;
- atomic WM/classifier checkpoint pairing;
- `WMCLS_COTRAIN_GLOBAL_STEPS` compatibility;
- classifier output-dimension inspection and BCE/CE alignment;
- the failure-imagined-RL requirement for warm checkpoints unless resuming;
- exact, distinct `CUDA_VISIBLE_DEVICES` validation against
  `manual_cotrain.ngpu`;
- cotrain-specific launch summary text.

Generic `--resume` remains implemented only once in `launchers.train` and keeps
the original run-root ownership semantics.

## Hydra Configuration

`configs/experiment/openvla_libero.yaml` declares the cotrain launch contract and
all static launch defaults. The configuration includes the contract target,
single-process execution, GPU count interpolation, LIBERO setup, and cotrain
environment variables. Other experiments remain on the no-op/default contract.

This preserves Hydra as the source of truth and makes the Python launcher consume
configuration rather than choose cotrain behavior.

## Public Surface and Compatibility

`scripts/experiments/cotrain/train.sh` delegates to
`dreamervla.launchers.train`. The documented `--config`, `--wm_ckpt`,
`--cls_ckpt`, `--resume`, raw Hydra overrides, checkpoint environment variables,
and cotrain dry-run behavior remain supported.

`dreamervla/launchers/cotrain.py` is deleted. It is not retained as a compatibility
wrapper because the repository documents shell entrypoints as the stable public
surface and retaining a second Python route would defeat the single-entrypoint
contract.

## Testing

Migrate cotrain launcher tests to `dreamervla.launchers.train.build_launch` and
prepend `--config openvla_libero`. Preserve tests for checkpoint pairing,
classifier inference, resume run-root reuse, environment compatibility, GPU
geometry, and public flag conflicts.

Add structural assertions that:

- the cotrain shell script calls only `dreamervla.launchers.train`;
- `dreamervla/launchers/cotrain.py` no longer exists;
- the generic launcher contains no experiment-name branch;
- the OpenVLA experiment selects the cotrain contract through Hydra;
- generic train-launcher tests continue to pass.

Run the focused launcher/config/script suites plus repository hygiene checks.

## Concurrent Worktree Safety

The current worktree contains unrelated runner and script-tree edits. The
implementation must stage only launcher-owned files and preserve all existing
unstaged changes. Where a required test or documentation file is already modified,
edit only non-overlapping assertions and inspect the combined diff before commit.
