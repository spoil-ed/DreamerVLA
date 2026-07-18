# Algorithm Content Polish Design

## Scope

This change hardens and documents every public algorithm family without changing
the accepted mainline objective. The current `openvla_libero` route remains
failure-conditioned, frozen-WM/classifier, imagined-only actor PPO. Its initial
condition remains `failed_episode_start`; the unfinished endpoint/KIR proposal is
not activated by this work.

The public algorithm families in scope are:

- DreamerV3 world-model pretraining and imagined actor-critic;
- LUMOS outcome PPO with sparse or probability reward;
- dense step and dense chunk PPO compatibility routes;
- the success verifier and two-hot critic contracts;
- eval-only TD-MPC latent MPC.

## Chosen Approach

Use contract-first hardening. Keep equations and defaults intact, but validate
their domains at Hydra startup and again at direct Python API boundaries. Add
focused numerical tests, replace benchmark-changing exception fallbacks with
explicit failures, and publish one algorithm reference whose route descriptions
match the effective configs.

This is preferred over documentation-only editing, which would preserve runtime
ambiguity, and over a wholesale PPO/cotrain rewrite, which would invalidate the
already verified training route without evidence that its objective is wrong.

## Architecture

`dreamervla.algorithms.validation` owns reusable, framework-light validation for
PPO and TD-MPC hyperparameters. `dreamervla.config.validate_cfg` applies it to the
top-level algorithm block and the actor/learner algorithm blocks before workers
spawn. Direct numerical helpers retain local guards so callers that bypass Hydra
receive the same fail-fast behavior.

Model-specific constructors validate invariants that configuration cannot know:
two-hot bin geometry, percentile bounds, non-empty return samples, and TD-MPC CEM
geometry. Reward builders validate tensor cardinality and finite scalar geometry
before scatter operations.

The runtime supports capability-based fallback only when a checkpoint genuinely
lacks `generate_action_head`. If that method exists and fails, evaluation raises a
contextual error rather than silently evaluating a different generation path.
Spatial-codec world models similarly fail at setup when their required image-token
mapping cannot be attached.

## Data and Error Flow

1. Hydra composes the experiment.
2. `validate_cfg` checks algorithm ranges and cross-field relationships.
3. Runner setup attaches required model interfaces; missing required capabilities
   stop setup with a chained exception.
4. Numerical helpers check direct-call invariants and execute unchanged equations.
5. Metrics and checkpoints keep their existing schemas.

Errors name the exact config path or API argument. No validation function selects
defaults or rewrites behavior.

## Documentation Contract

`docs/algorithms.md` becomes the public algorithm reference. It records each route,
its reward/value source, trainable parameters, equations, accepted ranges, and
whether it is mainline or supporting. `spec/04_complete_loop.md` remains the
command/data-flow authority and must agree with the manual-notes checkpoint rule:
replay is transient and is not serialized in cotrain checkpoints.

Stale package maps are corrected to the converged `models/embodiment` and
`algorithms/{actor,critic}` layout. Parameter documentation links to the algorithm
reference instead of duplicating unexplained knobs.

## Verification

Every behavioral change follows red-green TDD. Tests cover invalid/valid config
ranges, GRPO empty and malformed groups, PPO clipping geometry, reward tensor
shape/cardinality, two-hot and percentile domains, TD-MPC geometry, WM mapping
fail-fast, and action-head failure propagation. Existing golden and microbatch
equivalence tests prove valid-path numerics remain unchanged.

Final gates are focused algorithm/runtime tests, all unit tests, config composition
for every experiment, Ruff check/format, shell syntax, and `git diff --check`.

## Non-goals

- No KIR/endpoint selector rollout change.
- No new reward function, optimizer, loss term, or default hyperparameter.
- No edits to `third_party/`.
- No broad runner/worker file split unrelated to algorithm correctness.
