# Online Keyframe-Initialized Rollouts Design

## Objective

Add reference-style keyframe-initialized rollouts to DreamerVLA while preserving the
repository's online cotrain architecture. Failed real trajectories from cold-start
collection and current-step collection become KIR sources. DreamerVLA must not
copy RLinf's static `.npy` initialization pool.

## Semantics

For a failed real episode with final valid index `k`, KIR initializes imagination
at `episode[k]`. The world model receives the real observation-latent history ending
at `k`, the aligned real action history, the real proprio history required by the
active WM, and the episode's language/task conditioning. The next policy action is
therefore evaluated locally after the failure-near keyframe.

Ordinary initialization remains episode-start initialization. Hydra explicitly
controls the mixture between ordinary and KIR anchors. If KIR is requested but no
failed episode is available for a requested task, sampling falls back to an ordinary
anchor rather than starving WM rollout.

## Data Flow

```text
cold-start collected episodes ─┐
                               ├─> OnlineReplay records
current-step real episodes ────┘        │
                                        ├─ ordinary: episode start
                                        └─ KIR: failed episode endpoint
                                                   │
                                      aligned latent/action/proprio history
                                                   │
                                              ReplayWorker
                                                   │
                                repeat each anchor for one policy group
                                                   │
                                        WM trajectory EnvWorker
                                                   │
                                     LatentWorldModelEnv.reset()
                                                   │
                                          imagined rollout
```

## Ownership

- `OnlineReplay` owns anchor selection and extraction of aligned episode-local
  histories.
- `ReplayWorker` exposes the sampler without choosing defaults.
- The WM EnvWorker derives history length from the instantiated world model, requests
  initial conditions, repeats anchors by the configured policy group size, and
  installs them on each environment slot.
- `LatentWorldModelEnv` and the chunk world model own construction of the exact
  recurrent imagination state from supplied history.
- Hydra owns whether KIR is enabled and the ordinary/KIR mixture ratio.

## Exact Anchor Contract

The replay response keeps current-condition keys for compatibility and adds history
keys used by stateful chunk world models:

- `obs_embedding`: keyframe/current latent.
- `obs_embedding_history`: real latent frames ending at the anchor.
- `action_history`: real actions aligned to the returned latent history.
- `lang_emb`: task language embedding.
- `proprio`: keyframe/current proprioception.
- `proprio_history`: real proprio frames aligned to latent history.
- `is_kir`: whether the selected anchor is a failed endpoint.
- `anchor_step`: episode-local anchor index.

Short histories are left-padded within the same episode using the first observation
and proprio state. Missing pre-history actions are padded with the model's neutral
action value; no history may cross an episode boundary.

## Grouping

Each sampled anchor is repeated for the active policy optimization group so all
actions within a comparison group start from the same real context. Group repetition
applies to current keys, history keys, and KIR metadata together.

## Metrics

WM refresh reports the number/fraction of KIR anchors and mean anchor step using the
existing `env/` metric route. Cosine similarity remains a WM evaluation metric and is
unrelated to KIR selection.

## Failure Handling

- Missing failed candidates: fall back to ordinary initialization for that task.
- Missing required latent/language/proprio/action fields: fail with a precise replay
  contract error rather than silently substituting unrelated data.
- Inconsistent history dimensions: fail before world-model rollout.
- Non-stateful WMs continue to consume the current condition and ignore optional
  history fields through narrow capability checks.

## Tests

Unit tests will prove:

1. Failed-only endpoint selection and correct keyframe metadata.
2. Exact latent/action/proprio history alignment and episode-local padding.
3. Fallback to ordinary starts when no failed episode exists.
4. Group repetition preserves all history and metadata.
5. Chunk-WM initialization uses supplied real history/actions instead of repeated
   latent and zero actions.
6. Hydra composition exposes KIR behavior without shell defaults.
