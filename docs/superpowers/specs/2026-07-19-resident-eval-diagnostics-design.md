# Resident Cotrain WM/CLS Diagnostics Design

## Objective

Every resident policy evaluation in the isolated aggressive Dreamer experiment
must emit task success together with a read-only measurement of the frozen world
model and success classifier on the same physical LIBERO trajectories.

## Metric contract

The implementation reuses `dreamervla.runtime.cotrain_eval` without changing its
semantics:

- `eval/wm_closed_loop_cosine` is computed by seeding the WM with real history,
  recursively predicting the remainder of each trajectory, averaging cosine over
  the available horizon within each trajectory, then weighting trajectories equally.
- `eval/classifier_real_f1` and `eval/classifier_real_accuracy` classify each real
  trajectory using the checkpoint-owned threshold and episode success as ground truth.
- `eval/classifier_wm_f1` and `eval/classifier_wm_accuracy` apply the same classifier
  to the recursively predicted WM trajectory. These expose composition quality without
  conflating it with the requested real-trajectory classifier quality.

The existing detailed summary and auxiliary precision/recall/AUC metrics remain in
the returned metric payload. No threshold is recalibrated during evaluation.

## Data flow

`EvaluationEnvironmentWorker` retains completed eval episodes only until the runner
drains the current global step. `replay_write_enabled` remains false, so the eval path
cannot mutate replay. The runner sends that batch through the resident Actor's existing
no-gradient raw-image re-encoder, then passes the encoded batch to `LearnerWorker`.
The learner converts transitions to `EncodedEvalTrajectory`, runs the existing WM/CLS
diagnostic, and returns flat scalar metrics to `BaseRunner.log_metrics` at the same
global step as policy success.

This keeps ownership aligned with the mainline architecture: Actor owns the current
VLA encoder; Learner owns WM/CLS; Env owns physical trajectories; Runner coordinates.

## Experiment strength

Only `experiment=openvla_libero_aggressive` changes. It evaluates every global step,
uses actor LR `2e-6`, two PPO epochs, `kl_beta=0.005`, and a transaction rollback limit
of `0.05`. This is a fourfold LR increase from mainline with twice the update reuse,
while retaining an explicit trust-region rollback boundary. The original
`openvla_libero` experiment is untouched.

## Failure behavior

Each eval must produce exactly the configured trajectory count. Missing sidecars,
short trajectories, count mismatches, or inconsistent WM/CLS records fail the eval
instead of silently emitting partial or stale metrics. Eval batches are drained exactly
once, including the initial step-zero evaluation.
