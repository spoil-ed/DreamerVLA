# DreamerVLA History

This document is the consolidated project history for DreamerVLA. It keeps the
main storyline in one editable place while the older source notes live under
`docs/archive/history_sources/`.

## TODO

- [ ] Edit this section into the current working TODO.

## Short Version

DreamerVLA started from a simple question: can a Dreamer-style world model use a
VLA policy's internal action-relevant hidden state as its latent observation, and
then improve the policy through imagined rollouts?

The project moved through several interfaces before settling on the current
direction. Early notes compared Dreamer/TransDreamer dynamics, action-aware
world models, inverse dynamics, and shuffled-action diagnostics. The first
implementation line used RynnVLA/pi0 action hidden as the observation target for
a DreamerV3-style RSSM. Diagnostics then showed that the hidden target is not a
generic full transformer state: it is mostly a 5-step action-time structure with
redundant joint copies and a very low effective rank. That pushed the work away
from full hidden reconstruction as a generic objective and toward action-hidden
interfaces that are explicitly aligned with the VLA action head.

The latest mainline is the DINO-WM / WMPO-style route. It treats frozen RynnVLA
legacy full action hidden as a learned hidden-state space, trains a
Transformer-style hidden dynamics model and reward/success heads, and then uses
PPO/WMPO-style imagined rollouts inside that learned hidden environment to train
the final DreamerVLA actor.

## Storyline

### 1. Seed architecture: VLA state plus WM head

The earliest sketch separated a VLA action head from a world-model head:

```text
image_t, state_t -> state encoder -> h_t
text             -> text encoder  -> c

VLA head:
(h_t, c) -> action_pred

WM head:
(h_t, action_t, c) -> next_h_pred
```

This framed the whole project: the policy already has a useful representation,
so the world model should learn controllable dynamics in or around that
representation rather than learning from pixels alone.

### 2. Dreamer/TransDreamer and action-aware dynamics

The literature notes clarified that standard Dreamer-style models learn
dynamics through posterior/prior alignment, reconstruction, reward/continue
prediction, and imagined actor-critic training. TransDreamer follows the same
basic pattern: prior dynamics predict latent state from previous latents and
actions, while the posterior uses the real observation.

The important gap was action sensitivity. A model can reduce reconstruction
loss while still not caring enough about which action caused which transition.
The notes connected this to nearby ideas:

- WAM and inverse dynamics force latent transitions to preserve action
  information by predicting the action from state change.
- Iso-Dream separates controllable from uncontrollable dynamics.
- ENACT-style shuffled-action evaluation asks whether a model can distinguish
  correct action sequences from shuffled ones.
- A possible DreamerVLA variant would turn shuffled-action evaluation into a
  training negative: real-action transitions should be closer to the target than
  shuffled-action transitions.

This line did not become the final implementation by itself, but it set the
project's recurring test: a useful latent must be action-causal, not merely
predictive.

### 3. Repository and workspace consolidation

The repo was then cleaned into a more stable shape:

- source code under `src/`
- launch and diagnostic entry points under `scripts/`
- Hydra configs under `configs/`
- generated data and checkpoints under `data/`
- route-specific workspace classes exported through `src.workspace`

The public execution path became:

```text
scripts/*.sh
  -> python -m src.cli.train --config-name <config>
  -> src/workspace/<workspace>.py
  -> src/models/, src/algorithms/, src/dataloader/
```

This made it easier to compare routes without mixing old workspaces, configs,
and scripts.

### 4. First mainline: pi0 action-hidden DreamerV3

The first concrete mainline used pi0 action-query hidden states as the
observation for a DreamerV3-style RSSM:

```text
LIBERO obs + language + state
-> frozen RynnVLA/Chameleon backbone
-> pi0 action-query block
-> action_hidden [H, 1024]
-> flattened action-hidden sidecar
-> DreamerV3 RSSM posterior / transition
-> hidden reconstruction + reward + continue
```

The intended actor route was:

```text
DreamerV3 RSSM feature
-> hidden decoder reconstructs action hidden
-> Pi0ActionHiddenActor
-> VLA output projection
-> action
```

This route established the central interface problem. Dreamer can produce a
rollout-able latent state, but the VLA action head expects action-head-compatible
hidden context. The hard part is the bridge between those two spaces.

### 5. RynnVLA hidden interface: what should be reconstructed?

The RynnVLA encode scheme narrowed the question. Reconstructing arbitrary full
VLA hidden sequences is too expensive and semantically noisy: prompt tokens,
image tokens, state tokens, and action-context tokens are all mixed together.
Much of that sequence is not a Markov physical state and should not be forced
through a Dreamer latent.

The better target is action-context hidden: the part of the representation that
the action head actually uses. Several alternatives were considered:

- pooled hidden vectors: easy but loses token structure
- predicted pooled hidden: MSE can decrease, but actor distribution shift is
  severe
- full token hidden sequence: too large and unstable
- direct actor MLP: stable, but discards the VLA action-head prior
- compact action-context bottleneck: preferred conceptually, because it keeps
  the action-head interface while avoiding full hidden reconstruction

This clarified that hidden reconstruction loss alone is not a sufficient success
criterion. The reconstructed hidden must remain usable by the VLA action head.

### 6. Hidden target diagnosis and the v4-F lesson

The hidden-structure diagnostic on pi0 action hidden found that the nominal
`[5, 7, 1024]` target is highly redundant:

- the 35 tokens are statistically near-identical at the marginal level
- same-time tokens across the 7 joints are almost duplicate residual signals
- the flattened 35840-dimensional target has a very low effective rank
- the structure is closer to a 5-step time sequence than a 5-by-7 joint grid

This led to the v4-F `pi0_time_broadcast` decoder idea:

```text
RSSM feature
-> predict 5 time tokens
-> broadcast each time token across 7 joints
-> flatten back to [5, 7, 1024]
```

The main lesson was not just a decoder trick. It showed that the useful hidden
state has strong structure, and that treating every action token as independent
adds capacity in the wrong place.

### 7. Eval diagnostics: input alignment was OK, trained actor was not

A fixed diagnostic protocol was added to compare offline preprocessing and
online rollout. Each new checkpoint should answer:

1. Is online eval input identical to offline sidecar input?
2. Are action-hidden and latent distributions aligned?
3. Do the original VLA actor and the trained Dreamer actor produce usable
   actions?

For the documented baseline, online/offline `obs_embedding` statistics were
almost perfectly aligned and original-live versus original-reconstructed actions
were close enough to show that the reconstructed hidden was not the primary
failure. The trained actor path, however, pushed actions toward saturated fixed
values.

This shifted attention from pure hidden reconstruction toward actor anchoring,
reference-policy constraints, reward modeling, and rollout stability.

### 8. DINO-WM route: hidden dynamics instead of RSSM reconstruction

The current method route moves from DreamerV3 RSSM reconstruction toward a
DINO-WM-style hidden dynamics model. The latest canonical hidden is legacy full
action hidden:

```text
u_t in R^{5 x 7 x 1024}
e_t = flatten(u_t) in R^{35840}
```

The data path is:

```text
LIBERO HDF5 demonstrations
-> no-op filtering / reward preprocessing
-> frozen RynnVLA legacy action-hidden sidecar
-> DINO-WM hidden dynamics and reward model
```

The DINO-WM predicts future hidden states under action conditioning:

```text
(e_{t-H+1:t}, a_{t-H+1:t}) -> e_{t+1}
```

It restores the `[35, 1024]` token structure, appends action tokens, uses a
frame-block causal Transformer, and trains with hidden MSE, cosine loss,
open-loop rollout loss, reward loss, and optional success-to-go auxiliary loss.

This route is not standard DreamerV3 RSSM. It is a learned hidden-space
environment built around VLA action-hidden states.

### 9. PPO/WMPO-style actor training

The policy optimization route uses DINO-WM as the imagination environment:

```text
offline batch hidden
-> actor samples action
-> DINO-WM predicts next hidden
-> reward / success-return head scores the transition
-> PPO-style update trains the actor
```

The actor still uses the VLA action-head prior through a legacy
action-hidden-compatible actor. PPO updates are constrained by KL and
BC-to-reference losses so the policy does not immediately drift into saturated
or meaningless actions.

The final eval checkpoint is the DreamerVLA policy checkpoint, not the DINO-WM
world-model checkpoint. The real LIBERO environment is used only at final
rollout evaluation.

### 10. WMPO full reproduction plan

The newest reproduction plan decomposes the route into five phases:

1. chunk-aware Rynn-DINO-WM with a 5-step prediction interface
2. LIBERO simulated rollout data collection for classifier training
3. latent success classifier and threshold tuning
4. `dino_wmpo_chunk_step`, the PPO/WMPO loop with correct reward placement
5. full PPO run and real-sim eval gating

During execution, classifier experiments showed that naive classifier training
plateaued around weak F1, and a WM-replay classifier dataset improved the path
but still needed better reward data or labeling. The important conclusion is
that the WMPO route depends not only on hidden prediction quality, but also on a
non-degenerate reward/success signal for imagined rollouts.

## Current Mainline

The current story to keep in mind is:

```text
Frozen VLA action hidden
-> learned DINO-WM hidden dynamics
-> reward / success-return model
-> PPO/WMPO-style imagined policy optimization
-> single-process LIBERO rollout eval
```

The important contracts are:

- action head type: `legacy` for the latest DINO-WM route
- hidden source: `action_query`
- prompt style: `vla_policy`
- history: `2`
- include robot state: `true`
- rotate images 180 degrees: `true`
- action horizon / action steps: `5`
- hidden dimension: `35840`

Do not mix legacy full action hidden with pi0-query `[5, 1024]` hidden, and do
not evaluate with an online input contract that differs from sidecar generation.

## Open Questions

- Which action-hidden target is best for the final actor interface: full legacy
  hidden, compact action-context tokens, or a chunk-aware hidden abstraction?
- How strong must the DINO-WM open-loop loss be before PPO rollouts become
  useful rather than self-confirming?
- What reward/success model produces a non-degenerate training signal for WMPO?
- How much KL/BC anchoring is needed to preserve the VLA action prior while
  still allowing policy improvement?
- Which diagnostic should be the final gate before expensive LIBERO rollout:
  hidden cosine, action agreement, reward calibration, or real-sim success?

## Archived Source Map

Original notes moved under `docs/archive/history_sources/`:

- `architecture.md`: first VLA-head / WM-head sketch.
- `1.md`: Dreamer, TransDreamer, action-aware WM, and shuffled-action notes.
- `repository_structure.md`: repo layout, workspace API, config/data policy.
- `wm_training_routes.md`: pi0 action-hidden DreamerV3 route and retained
  baselines.
- `rynnvla_encode_dreamervla_scheme.md`: RynnVLA hidden interface design and
  reconstruction-target analysis.
- `hidden_token_structure_report.md`: measured structure of the action-hidden
  reconstruction target.
- `dreamervla_eval_diagnostic_protocol.md`: offline/online and actor-path
  diagnostic protocol.
- `dreamervla_dino_wm_ppo_method.md`: latest DINO-WM to PPO/WMPO method
  description.
- `superpowers/`: implementation plans and design specs that record the
  step-by-step development history.
