# OpenVLA Staged Cotrain Design

**Date:** 2026-07-13
**Status:** Approved for implementation

## Objective

Replace the current hidden-token replacement policy with the original OpenVLA-OFT
policy boundary, and make each online global step a causal staged transaction:

1. collect a small current-policy real batch;
2. update only the VLA encoder from successful real trajectories;
3. re-encode the whole batch in the new latent space;
4. update and recalibrate the world model and success classifier in that space;
5. generate imagined trajectories with those latest models;
6. update the original VLA action decoder with PPO;
7. evaluate and checkpoint the complete transaction.

The initial policy remains a fixed monitoring anchor. The optimization trust region
is between the policy at the start of the global step and the policy after both the
encoder-SFT and imagined-PPO phases.

## Policy Boundary

The trainable policy remains OpenVLA-OFT:

```text
image / text / proprio
        |
OpenVLA vision backbone + projector (encoder E)
        |
projected visual tokens z: [token_count, token_dim]
        |
concat with checkpoint-defined text / proprio sequence
        |
original OpenVLA language model + OFT action decoder (actor A)
        |
action-token distribution -> robot actions
```

For the current checkpoint, the projected visual sidecar is `[256, 4096]`. Those
numbers are metadata derived from the VLA checkpoint and input scheme; they are not
hard-coded architecture constants. Text and optional proprio use the checkpoint's
native multimodal construction. They share the 4096 feature width and are concatenated
along the sequence axis where the upstream OpenVLA implementation expects them.

The mainline must not construct a random Transformer decoder or learned action-query
bank between projected tokens and action tokens. The existing
`LatentToOpenVLAHiddenStateActor` remains available only to isolated legacy/frozen
feasibility recipes until those recipes are explicitly migrated; it is not the
online-cotrain policy.

The same full policy module has two forward paths:

- **raw path:** pixels are passed through `E`, then through `A`; used by real rollout
  and successful-trajectory SFT;
- **latent path:** projected tokens supplied by the world model enter the same `A`;
  used by imagined rollout and PPO.

Both paths retain the native prompt, attention mask, proprio handling, action-token
positions, token bins, and action unnormalization defined by the OpenVLA-OFT
checkpoint. Actor training evaluates the exact sampled action-token IDs rather than
approximately inverting continuous actions.

## Three Policy States and One Trust Region

- `pi_initial`: immutable initial VLA, used only for monitoring and regression
  comparison.
- `pi_old`: full VLA snapshot at the start of a global step.
- `pi_new`: policy after encoder SFT and actor PPO in that global step.

There is one cumulative `pi_old -> pi_new` KL budget. Encoder SFT and actor PPO consume
the same budget; the allowance is not reset between phases. The two updates are
committed at causal barriers instead of being one unsafe cross-component atomic
transaction:

1. encoder SFT is measured on the full real-path action-token distribution and is
   accepted only when it fits the total budget; otherwise the policy and encoder
   optimizer are restored before re-encoding;
2. after an accepted encoder update, WM/CLS are fitted in that new space;
3. actor PPO may consume only `budget - accepted_encoder_KL`; if it exceeds the
   remainder, the policy and actor optimizer return to the post-SFT state.

This makes the final accepted sum bounded by one budget while preserving WM/CLS
alignment with the committed encoder. Real-path and imagined-path KL values are
reported separately and as their effective cumulative sum. `pi_initial` is not
included in the optimization loss.

## Global-Step Transaction

### 1. Current-policy real collection

`pi_old` collects exactly 32 completed real trajectories across the selected LIBERO
tasks. The step-local batch retains raw images, task text, checkpoint-required
proprio/state, sampled action-token IDs, executed actions, reward, termination, and
success. The batch is replaced at the next global step; there is no cross-step replay
for encoder, WM, or classifier updates. Each real slot is reset at the global-step
boundary, and any unfinished previous-policy episode is discarded rather than relabeled
under the new step. Each slot/rollout epoch stops after its first terminal episode; an
early LIBERO success therefore cannot make the nominal 32-trajectory budget overflow.

### 2. Encoder-only successful self-imitation

Only successful real trajectories are used. The original actor remains in the
differentiable graph but its parameters are frozen; a token-level SFT loss updates the
vision backbone and projector with one uniformly low learning rate. There is no
latent-anchor loss and no classifier pseudo-label. If the real batch has no successful
trajectory, encoder SFT is skipped and this is reported.

### 3. Re-encode in the new space

All 32 trajectories, including failures, are re-encoded with the updated encoder.
The resulting transition batch contains the new projected tokens together with the
same actions, language/proprio conditioning, reward, terminal markers, and success
labels. This barrier is mandatory: no old-encoder latent may enter the WM/CLS update.

### 4. WM and classifier fitting

WM and classifier are updated on only the current re-encoded batch, using multiple
optimizer steps over multiple epochs with early stopping. Their parameters and
optimizer states persist between global steps, but their training examples do not.

The classifier threshold is recalibrated from a step-local calibration split. If the
split lacks either class, the previous threshold is retained and the skipped
calibration is reported.

### 5. Latest-model imagined rollout

The latest WM and classifier states are synchronized to WM environments. Imagined
episodes start from current-step re-encoded histories. The rollout policy uses the
same OpenVLA actor with the native text/proprio conditioning. WM rollouts are generated
only after the WM/CLS update barrier.

### 6. Actor-only imagined PPO

The updated encoder is frozen. PPO trains the original OpenVLA actor only, using
imagined trajectories. Real trajectories do not enter PPO. The PPO objective keeps
the existing clipping/advantage machinery and participates in the same step-level KL
budget established before encoder SFT.

### 7. Eval and checkpoint

After the actor update, `pi_new` is the policy for the next global step. The checkpoint
contains the complete VLA, WM, classifier, all optimizer states, classifier threshold,
global step, component versions, and the effective KL metrics. Periodic VLA evaluation
loads the full VLA checkpoint rather than combining a frozen base VLA with a replacement
bridge.

## World-Model Semantics

The current `ChunkAwareWorldModel` uses history `H=3`, action chunk `K=8`, and a
four-chunk training rollout (32 physical steps). Training keeps truncated closed-loop
backpropagation at this 32-step horizon. Its loss continues to include one-step/token
MSE and multi-chunk rollout MSE; cosine is promoted to a first-class reported metric
and can be assigned a small nonzero configured weight.

Inference must carry the model-returned rolling hidden history and action history from
one chunk to the next. Reconstructing every next input from only the final latent and
zero/repeated history is a train/use mismatch and is prohibited.

Full-trajectory autoregression is an evaluation protocol, not a training BPTT horizon.

## Fixed 100-Trajectory Evaluation

Every configured evaluation interval collects 10 tasks by 10 fixed initial states
with `pi_new`. These trajectories are read-only evaluation data: they never train a
component and never calibrate the classifier threshold.

The report contains:

- VLA real success rate, overall and per task;
- WM fully closed-loop trajectory evaluation, initialized from the first true `H=3`
  observations and then recursively predicted for the entire recorded action sequence;
- equal-weight per-trajectory token MSE, cosine similarity, and horizon error curves;
- classifier metrics on real `pi_new` latents: trajectory-level success F1, precision,
  recall, accuracy, confusion counts, positive rate, predicted-positive rate, plus
  auxiliary chunk/window metrics;
- the same classifier metrics on fully closed-loop WM latents;
- PR-AUC and ROC-AUC when both classes exist;
- the frozen step-local classifier threshold used for every evaluation result.

## Failure and Resume Semantics

Each global step is checkpointed only at a completed transaction boundary. Resume
restores the full component and optimizer bundle from the last boundary. Optional
Hydra resume keys injected by launchers use `++manual_cotrain.resume_ckpt=...`, so
recipes that do not predeclare the field compose correctly.

If a phase produces no usable examples, that phase is explicitly skipped and logged;
the runner does not substitute stale examples from a prior global step. A KL budget
violation rolls back the violating phase and its optimizer state. Encoder rejection
happens before re-encoding and WM/CLS fitting; PPO rejection restores the accepted
post-SFT policy. No rollback crosses a WM/CLS latent-space update barrier.
