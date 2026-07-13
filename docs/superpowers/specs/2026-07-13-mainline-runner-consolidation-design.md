# Mainline Runner Consolidation Design

## Status

Approved design for converging DreamerVLA on one production workflow:

```text
collect real rollouts
-> train missing World Model / Success Classifier components
-> cotrain World Model + Success Classifier + VLA
-> evaluate in real LIBERO
```

The cotrain implementation is the current staged Ray route. Frozen-policy and
alternate online-cotrain experiments are not part of the target architecture.

## Goals

- Keep only runner classes needed by the production workflow.
- Give every public runner one complete, role-based name.
- Preserve independent World Model and Success Classifier training.
- Reuse supplied component checkpoints and train only missing components.
- Keep all cotrain models resident on their assigned GPUs while executing stages
  serially.
- Evaluate the resident policy without restarting Ray or reloading the models.
- Report separate progress for real collection, VLA real SFT, WM/CLS training,
  imagined collection, VLA PPO, and evaluation.
- Preserve both native PyTorch and Hugging Face checkpoint boundaries used by the
  mainline.

## Public Runner Surface

The public runner surface contains exactly these concrete runners:

1. `RolloutCollectionRunner`
2. `WorldModelTrainingRunner`
3. `SuccessClassifierTrainingRunner`
4. `CotrainRunner`
5. `LIBEROVLAEvaluationRunner`

`BaseRunner` remains the abstract lifecycle and artifact owner. Internal helpers
are grouped by collection, component training, cotrain, and evaluation; they are
not exported as additional runners.

The following runner families are removed or absorbed:

- frozen-model policy training;
- joint/follow-up DreamerVLA experiments;
- legacy synchronous online cotrain;
- alternate Ray online cotrain;
- the manual-cotrain public name;
- pipeline runners that combine orchestration with model training;
- standalone runner aliases retained only for compatibility.

No compatibility proxy is retained for deleted runner names. Hydra experiments,
tests, scripts, and documentation move to the five canonical targets in the same
change.

## Pipeline Routing

The Python launcher owns stage orchestration. Each stage remains one runner job.

| World Model checkpoint | Classifier checkpoint | Required stages before cotrain |
|---|---|---|
| missing | missing | collect, World Model training, classifier training |
| present | missing | collect if no valid collected data exists, classifier training |
| missing | present | collect if no valid collected data exists, World Model training |
| present | present | none |

After both component checkpoints exist, every route enters `CotrainRunner` and
then performs periodic real-LIBERO evaluation. A supplied checkpoint initializes
the component; it does not freeze it during cotrain.

Collection artifacts are reused only after their manifest and hidden-sidecar
metadata pass the existing task/checkpoint shape contracts. Missing or mismatched
artifacts trigger collection rather than silent reuse.

## Independent Component Training

`WorldModelTrainingRunner` trains and checkpoints only the World Model and its
optimizer. `SuccessClassifierTrainingRunner` trains and checkpoints only the
Success Classifier, calibrated threshold, and its optimizer. Both consume the
canonical collected reward/hidden dataset contract and can be launched directly.

The combined launcher executes these jobs serially. It does not require one
component to be retrained when a valid checkpoint for that component was supplied.
Their outputs use the same component-loading protocol accepted by `CotrainRunner`.

## Cotrain Causal Transaction

One `CotrainRunner` global step is a fixed transaction:

1. Synchronize the step-entry VLA (`pi_old`) from `ActorGroup` to `RolloutGroup`.
2. Collect exactly the configured number of complete real trajectories with the
   current policy: 32 in production and 8 in debug.
3. Train the VLA vision encoder/projector with successful real action-token labels.
4. Re-encode all current-step real trajectories in the accepted encoder space.
5. Replace replay with only this current-step batch.
6. Train the World Model and Success Classifier in `LearnerGroup`, preserving model
   and optimizer state across global steps while not retaining stale training
   trajectories.
7. Synchronize the updated World Model, Success Classifier, and calibrated threshold
   to `WorldModelEnvGroup`.
8. Synchronize the accepted post-SFT VLA to `RolloutGroup`.
9. Generate exactly the configured number of imagined trajectories: 1024 in
   production and 256 in debug. The Success Classifier supplies their reward.
10. Compute advantages/returns and train the full VLA actor with PPO using imagined
    trajectories only.
11. Apply one shared KL budget across encoder SFT and actor PPO. Roll back the
    corresponding policy/optimizer transaction when its remaining budget is
    exceeded.
12. Save the complete trainable state and run read-only evaluation at the configured
    interval.

World Model / Success Classifier optimization and Actor optimization never overlap.
Real trajectories never enter PPO, and imagined trajectories never train the
World Model or Success Classifier in the same transaction.

## Ray Residency and GPU Placement

The eight-GPU production topology is fixed and explicit:

| Role | Placement | Residency |
|---|---|---|
| `ActorGroup` | GPU 0-7, one FSDP rank per GPU | entire cotrain run |
| `RolloutGroup` | GPU 0-7, one no-grad policy replica per GPU | entire cotrain run |
| `LearnerGroup` | GPU 0, World Model + Success Classifier | entire cotrain run |
| `WorldModelEnvGroup` | GPU 1-7 | entire cotrain run |
| `RealEnvironmentGroup` | CPU OSMesa, paired with rollout rank 0 | entire cotrain run |
| `EvaluationEnvironmentGroup` | CPU OSMesa, 25 envs by default | entire cotrain run |
| `ReplayGroup` | CPU | entire cotrain run |

Workers and models are launched once. Cotrain uses barriers to activate these
phases serially:

```text
real rollout
-> VLA real SFT
-> World Model / Success Classifier training
-> imagined rollout
-> VLA PPO
-> optional evaluation
```

Inactive workers keep their models resident but perform no optimizer or environment
work. Stage transitions synchronize only updated state; they do not destroy and
reload models.

## Resident Evaluation

Periodic evaluation is executed inside the persistent cotrain Ray job:

- `ActorGroup` synchronizes the accepted VLA to the resident `RolloutGroup`.
- CPU `EvaluationEnvironmentGroup` runs the fixed 10-task, 100-episode protocol.
- Evaluation observations/actions pass through a dedicated channel to the resident
  `RolloutGroup`.
- Evaluation is read-only: no replay writes, optimizer steps, threshold calibration,
  or training RNG mutation.
- Training resumes only after the evaluation barrier completes.

This replaces subprocess segmentation that stopped Ray, loaded a checkpoint into a
new policy, evaluated it, and reloaded all training models. The standalone
`LIBEROVLAEvaluationRunner` remains available for arbitrary saved checkpoints and
final offline evaluation.

## Progress Reporting

Rank 0 reports independent progress lines, refreshed at the configured console
interval (five seconds by default):

```text
cotrain-real-rollout/<step>
cotrain-vla-real-sft/<step>
cotrain-wmcls-training/<step>
cotrain-imagined-rollout/<step>
cotrain-vla-ppo/<step>
eval/<step>
cotrain
```

Required status fields:

- real rollout: completed/target trajectories, environment chunks, successes;
- VLA real SFT: epoch, batch, optimizer step, loss, effective KL;
- WM/CLS training: completed/target learner updates, WM loss, classifier loss and
  F1, threshold, early-stop status;
- imagined rollout: completed/target trajectories, chunks, classifier-positive rate;
- VLA PPO: global batch, micro-batch, optimizer step, loss, KL, clip fraction;
- evaluation: completed/target episodes, successes, success rate, environment
  chunk throughput;
- cotrain: completed/target global steps and current phase.

Real and imagined rollout totals are never aggregated into the same bar. WM/CLS and
Actor training never share one training progress total. A stalled phase retains its
last visible status for diagnosis.

## Checkpoints and Resume

Production saves every 10 global steps. Debug runs for 10 global steps and saves and
evaluates every step, with 8 real and 256 imagined trajectories. Other debug values
inherit production defaults.

Debug limits are enforced by the resolved `CotrainRunner` configuration before the
driver loop starts; they are not launcher-only hints. A debug run must terminate after
exactly 10 accepted global steps even when the shell entrypoint or base experiment
declares a larger production target. Resume and evaluation orchestration must not
replace that cap with a larger target.

A cotrain checkpoint contains the complete VLA, World Model, Success Classifier,
classifier threshold, all active optimizers, global step, and required replay/sampling
metadata. Component checkpoints remain loadable as native PyTorch state. VLA export
and standalone evaluation retain the Hugging Face checkpoint path required by the
OpenVLA-OFT boundary.

Resume restores one accepted transaction boundary. It must not resume from a
partially completed real, learner, imagined, PPO, or evaluation phase.

## Failure Handling

- A worker failure aborts the current global-step transaction and leaves the last
  accepted checkpoint unchanged.
- A rollout timeout reports the phase-specific progress snapshot and worker ranks.
- Missing component checkpoints route to independent training; malformed or
  incompatible checkpoints fail before Ray workers launch.
- Evaluation failure fails the step acceptance barrier and does not silently continue
  training.
- Weight synchronization is versioned at step-entry, post-SFT, post-learner, and
  post-PPO boundaries.

## Verification

The consolidation is complete only when:

- the five public runner targets compose through Hydra;
- removed runner names have no source, config, script, test, or documentation refs;
- all four checkpoint-presence routing cases are tested;
- current staged cotrain causality tests pass under the `CotrainRunner` name;
- placement tests prove the fixed eight-GPU residency map;
- no Actor and Learner optimizer operation overlaps;
- periodic eval reuses resident rollout workers and does not launch a second VLA;
- all six phase progress streams advance monotonically and finish at their own totals;
- native PyTorch component restore and Hugging Face VLA evaluation/export both pass;
- debug and production schedules match their declared budgets.

The final acceptance test is a real local eight-GPU debug run with
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. It must launch all resident Ray groups, stop
after global step 10, collect 8 real and 256 imagined trajectories per accepted step,
save every step, evaluate every step, and expose all phase-specific progress streams.
Dry-run composition or a mocked worker test does not replace this acceptance run.

## Non-Goals

- Frozen World Model or classifier policy-only RL.
- Alternate synchronous cotrain implementations.
- Compatibility aliases for removed runner names.
- Multi-node Ray placement.
- Concurrent WM/CLS and Actor optimization.
- Replacing the OpenVLA-OFT action or checkpoint boundary.
