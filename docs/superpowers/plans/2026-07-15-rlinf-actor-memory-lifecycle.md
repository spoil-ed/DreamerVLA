# RLinf-Aligned Actor Memory Lifecycle Implementation Plan

> Execute continuously with test-driven development. Preserve unrelated dirty
> worktree changes and never edit the sibling RLinf or `third_party` trees.

**Goal:** Make consecutive Dreamer imagined-RL steps memory-safe by aligning
Actor/Rollout/WM residency, FSDP wrapping, and online weight synchronization
with RLinf while preserving checkpoint/resume behavior.

**Architecture:** Add generic phase-offload operations to DreamerVLA's FSDP
strategy, expose OpenVLA wrap capabilities at the model boundary, replace
online full-state patch export with atomic bounded buckets, and enforce phase
ordering in `DreamerRunner`/`CotrainRunner`. Full state remains restricted to
checkpoint operations; FSDP KL rollback snapshots only each rank's local
parameter, buffer, and optimizer shards.

**Tech stack:** Python 3.11, PyTorch FSDP1, Ray, Hydra/OmegaConf, pytest, Ruff.

## Task 1: Pin the FSDP and model capability contract

**Tests:**

- Extend `tests/unit_tests/test_fsdp_model_manager.py`.
- Extend `tests/unit_tests/test_fsdp_strategy.py`.
- Add focused OpenVLA policy capability tests without loading a checkpoint.

**Implementation:**

- Add FSDP settings for `sharding_strategy`, `forward_prefetch`,
  `backward_prefetch`, `limit_all_gathers`, and model-provided auto wrapping.
- Pass `device_id` and `FULL_SHARD` to FSDP1.
- Add OpenVLA wrap-target and gradient-checkpointing delegation methods.
- Enable activation checkpointing in the Dreamer mainline config.

**Verification:** Run the three focused test modules and Ruff on touched files.

## Task 2: Pin and implement Actor phase offload

**Tests:** Extend `tests/unit_tests/test_embodied_fsdp_actor.py` with probes for:

- init offload;
- parameter-only sync load;
- training parameter/optimizer load;
- normal and exceptional cleanup;
- lifecycle-aware checkpoint, optimizer export, and KL rollback.

**Implementation:**

- Add FSDP1 parameter/gradient onload and offload methods based on RLinf.
- Add optimizer state onload/offload methods.
- Add Actor residency flags, state transitions, CUDA metrics, and cleanup.
- Enable `actor.train_cfg.enable_offload` in the Dreamer mainline.
- Keep `fsdp.cpu_offload=false`.

**Verification:** Run the complete Actor unit-test module.

## Task 3: Replace online full-state patch export

**Tests:**

- Add atomic bucket-store tests.
- Prove online Actor sync uses sharded export rather than `state_dict()`.
- Prove Rollout applies CPU buckets without constructing another full snapshot.
- Prove version metadata is invisible until all buckets are published.

**Implementation:**

- Add sharded state export to the FSDP strategy.
- Implement a versioned atomic bucket syncer with configured bucket size.
- Materialize one bucket at a time on Actor rank zero.
- Apply buckets directly to the CPU-resident Rollout model.
- Start Rollout receive work before waiting for Actor send completion.

**Verification:** Run all weight-sync and resident-eval unit tests.

## Task 4: Align frozen WM environment residency

**Tests:** Extend WM environment worker tests to prove:

- frozen WM/classifier load before imagined interaction;
- both offload in `finally` after success or failure;
- component-state refresh works while offloaded.

**Implementation:**

- Add optional phase offload to `LatentWorldModelEnv`/`WMEnvWorker` through the
  existing config-selected environment capability.
- Enable it only for the frozen Dreamer route.
- Clear CUDA memory after interaction and before Actor training.

**Verification:** Run focused WM env and stage-order tests.

## Task 5: Preserve runner, KL, checkpoint, and resume contracts

**Tests:**

- Move the Dreamer PPO transaction assertion to after imagined rollout.
- Assert Actor/Rollout/WM phase order and residency checks.
- Round-trip manual checkpoint/resume with offloaded Actor state.
- Preserve policy hashes, optimizer state, RNG by rank, and run root.

**Implementation:**

- Move the failure-imagined-RL KL transaction start to immediately before PPO.
- Add lifecycle-aware full-state export wrappers with `finally` restoration.
- Keep the current checkpoint schema and public runner APIs.

**Verification:** Run cotrain stage-order, resume, DreamerRunner, and launcher
tests.

## Task 6: Integrated verification and commit

- Run focused Actor/FSDP/sync/WM/cotrain suites.
- Run the broader unit-test selection affected by configuration.
- Run gated Ray/CUDA smoke tests when the required hardware/checkpoint exists;
  otherwise report the explicit skip condition.
- Run Ruff checks and formatting checks for touched Python files.
- Run `bash -n` for any touched shell launchers.
- Run `git diff --check`.
- Review `git status`, stage only owned files, and create a signed Conventional
  Commit.
