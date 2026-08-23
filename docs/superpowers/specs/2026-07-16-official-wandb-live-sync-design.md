# Official W&B Live Sync Design

## Goal

Replace DreamerVLA's repository-owned offline W&B uploader with the official
`wandb beta sync --live` command for the shared-filesystem deployment where GPU
workers cannot reach W&B and a CPU host can read the run directory and access the
network.

## Scope

- Delete `dreamervla/launchers/wandb_sync.py`.
- Delete `scripts/utils/wandb_sync.sh`.
- Delete the launcher's dedicated unit tests.
- Document the official live command in active README, config, script-registry,
  and experiment-tutorial documentation.
- Keep W&B offline logging, stable run IDs, resume behavior, and canonical
  `${training.out_dir}/wandb` layout unchanged.
- Keep historical design and implementation-plan documents unchanged as records
  of earlier repository decisions.

## User workflow

The GPU training command continues to select offline mode:

```bash
logger=tensorboard_wandb runner.logger.wandb_mode=offline
```

After the GPU process has created `${OUT_DIR}/wandb/offline-run-*`, the networked
CPU host authenticates and tails the canonical W&B directory:

```bash
wandb login
wandb beta sync --live "${OUT_DIR}/wandb"
```

The CPU host uses W&B 0.24.1 or newer. Version 0.24.0 is excluded because it was
yanked; the repository's inspected environment uses 0.24.2. The live command
normally exits after the writer finishes cleanly. If the writer crashes, the beta
command may wait indefinitely; the operator stops it and performs a final
non-live sync:

```bash
wandb beta sync "${OUT_DIR}/wandb"
```

For legacy directories that contain multiple unrelated run IDs, the operator
passes the exact active `offline-run-*` directory instead of the shared parent.

## Tests

Repository hygiene tests assert that the custom launcher, wrapper, and dedicated
test file do not exist. Active documentation must contain the official live
command and must not reference the removed wrapper. Existing metric-logger tests
continue to cover offline layout and stable run identity.
