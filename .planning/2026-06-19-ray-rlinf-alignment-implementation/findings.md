# Findings: Ray RLinf Alignment Implementation

## Current Worktree Findings
- Prior dirty work already started manual precision, FSDP, collective, hardware, and lazy runner changes.
- The main P0 gap was still real DreamerVLA cotrain updates in `LearnerWorker`.

## Existing Real Update Functions
- WM: `dreamervla.algorithms.dreamervla.world_model_pretrain_step`.
- Classifier: `dreamervla.runners.online_dreamervla.online_classifier_update_step`.
- RL: `dreamervla.algorithms.ppo.dino_wmpo_outcome_step`.

## Implemented Alignment
- `LearnerWorker(mode="dreamervla_cotrain")` routes `wm`, `classifier`, `rl`, and combined `cotrain` phases through the existing DreamerVLA update functions.
- `ReplayWorker` exposes classifier-window APIs needed by Ray learner classifier updates.
- `OnlineCotrainRayRunner` selects `cotrain` phase for real DreamerVLA learner mode and reports `train/learner_updates` while preserving `train/ppo_updates`.
- Manual precision, FSDP manager, collective weight syncer, hardware discovery, FlexiblePlacement, model registry, and precision/parallelism config groups are implemented with tests.

## Residual Scope
- Multi-node manager/node-affinity/dynamic-scheduler expansion remains a future scale-out layer.
- S5 exact numeric parity against `OnlineCotrainPipelineRunner` still needs a dedicated real-config parity fixture; current coverage proves the new real phase boundary and preserves synthetic Ray e2e.
