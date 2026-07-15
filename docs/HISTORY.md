# HISTORY

This document records shipped DreamerVLA work at a high level. Commit-level
detail lives in `git log`; architecture rules live in [AGENTS.md](../AGENTS.md).

- Last updated: 2026-07-11

## Training Entry

- Single grouped Hydra entry:
  `python -m dreamervla.train experiment=<name> task=<suite>`.
- Public runners use `setup() -> execute() -> teardown()`.
- Current release route:
  `collect rollouts -> seed replay -> warm up world model + classifier -> online cotrain -> eval`.
- Main launcher configs:
  `configs/scripts/coldstart_warmup_cotrain.yaml`,
  `openvla_onetraj_libero_cotrain`,
  `openvla_onetraj_libero_cotrain`,
  `wm_full_dataset_train`, and `eval_libero_vla`.

## Ray Backend

- Single-node Ray placement, worker groups, channels, learner workers, rollout
  workers, and weight sync live under `scheduler/`, `workers/`, and
  `hybrid_engines/`.
- `CotrainRunner` is the sole online-cotrain runner.
- Standalone WM/classifier runners own warmup and checkpoint handoff.

## Warmup And Replay

- `CollectRolloutsRunner` and `ColdStartRayCollectRunner` write reward and
  hidden HDF5 shards under `collected_rollouts/<suite>/{reward,hidden}`.
- `WorldModelTrainingRunner` and `SuccessClassifierTrainingRunner` consume the
  collected shards and write their split warmup checkpoints.

## Diagnostics

- WM overfit probes, collection completeness checks, install verification, and
  LIBERO eval utilities live under `dreamervla/diagnostics/` and
  `scripts/experiments/`.
