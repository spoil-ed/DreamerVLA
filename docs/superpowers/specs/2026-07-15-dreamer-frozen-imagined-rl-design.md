# Dreamer Frozen Imagined-RL Runner Design

## Goal

Make `experiment=openvla_libero` use a dedicated `DreamerRunner` that trains the
OpenVLA-OFT actor only from world-model-imagined latent trajectories. Encoder SFT,
world-model updates, and classifier updates remain disabled. Resolve the first PPO
micro-batch CUDA OOM without changing the retained `CotrainRunner` behavior.

## Preserved implementation

`dreamervla/runners/cotrain_runner.py` remains unchanged and continues to export
`CotrainRunner`. Its copied implementation in
`dreamervla/runners/dreamer_runner.py` is renamed to `DreamerRunner` and specialized
for frozen imagined RL. Existing full-cotrain behavior therefore remains available
through the original module and unchanged.

## Training semantics

The active route keeps the existing causal order:

1. Collect complete real LIBERO trajectories.
2. Append them to historical replay.
3. Select initial latent states from failed episodes.
4. Use the frozen world model and classifier to generate and score imagined latent
   trajectories.
5. Send only imagined trajectories to ActorGroup.
6. Run advantage computation and PPO on the native OpenVLA-OFT actor path.

The route uses `manual_cotrain.training_mode=failure_imagined_rl`,
`learner_updates_enabled=false`, and `staged_policy_update=false`. Consequently it
does not call encoder SFT/re-encoding, world-model updates, classifier updates, or
learner-to-environment update synchronization. No additional per-parameter gradient
switch is introduced: the PPO inputs begin after the encoder at latent state, and
the disabled update stages define the frozen component boundary.

## Placement and memory

ActorGroup continues to use all eight GPUs for FSDP. RolloutGroup retains its
existing inference-copy offload lifecycle. WMEnvGroup keeps the frozen world model
and classifier on the GPUs needed for imagination.

The frozen route still needs the checkpoint-owning LearnerGroup during setup and
checkpointing, but it never trains. `DreamerRunner` therefore places that worker on
the CPU instead of co-locating another world-model/classifier copy on Actor rank 0.
This removes the approximately 3.7 GiB non-actor CUDA resident observed in the OOM
trace while preserving the existing state-loading and checkpoint code.

The mainline actor micro-batch is reduced from 32 to 8. The global batch remains
16384, so optimizer-step semantics and effective batch size do not change; only the
number of accumulated forward/backward micro-batches changes. This bounds Llama
attention activation memory on every actor rank while retaining the existing PPO
implementation.

## Public route

`DreamerRunner` is exported from `dreamervla.runners`. The `openvla_libero`
experiment targets `dreamervla.runners.DreamerRunner`. Other experiments keep their
current targets.

## Validation and tests

Tests will prove that:

- the original `CotrainRunner` remains exported from its original module;
- `DreamerRunner` is exported and selected by `openvla_libero`;
- the Dreamer placement keeps all actor GPU ranks but assigns LearnerGroup no GPU;
- the active route keeps encoder, WM, and classifier update stages disabled;
- the configured actor micro-batch is 8 while the global batch remains 16384;
- the focused runner/config/stage-order tests and lint checks pass.
