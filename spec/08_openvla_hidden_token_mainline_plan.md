# OpenVLA-OFT Hidden-Token Mainline Implementation Record

## Goal

Use `hidden_token [256,4096]` as the only public OpenVLA-OFT observation
sidecar contract. Keep the discrete decoder's internal 56 action slots private
to action decoding; they are never a world-model observation or sidecar.

## Canonical Contract

```yaml
task:
  openvla_oft:
    hidden_token_dir: ${task.hdf5_dir}_oft_hidden_token_vla_policy_h1
    hidden_token:
      expected_obs_hidden_source: hidden_token
      token_count: 256
      token_dim: 4096
      wm_obs_dim: 1048576
```

The `4096` token width describes projected vision tokens before world-model
conditioning. Proprioception, language, and action features are concatenated
later; their internal model dimensions remain separate Hydra settings.

## Implemented Work

- Hydra task, experiment, runner, dataset, launcher, diagnostics, and test
  surfaces use the `hidden_token` role and `hidden_token_dir` path.
- Offline preprocessing writes `_oft_hidden_token_vla_policy_h1`, declares
  `obs_hidden_source=hidden_token`, stores `[T,256,4096]`, and atomically marks
  every completed demo and shard.
- Reward preprocessing atomically marks every completed demo and shard so its
  output immediately satisfies the replay/training validator.
- Collection and offline replay validate metadata, file sets, demo sets,
  completion markers, frame alignment, and token shape before training.
- A narrow read-only migration adapter accepts the known historical projected
  sidecars only when their HDF5 attributes prove the same `[256,4096]` payload
  and removed action/actor payloads are disabled. New artifacts never emit the
  historical source value.
- Synthetic 56-token sidecars remain rejected.

## Verification Gates

1. Compose classifier, world-model, frozen-policy RL, sync-cotrain, and
   Ray-cotrain Hydra recipes.
2. Run focused preprocessing, sidecar-schema, configuration, runner, and
   repository-hygiene unit tests.
3. Confirm the retired public filenames and Hydra keys are absent outside the
   explicit migration regression fixture.
4. Confirm `third_party/` is untouched.

Real LIBERO/OpenVLA training is intentionally excluded from repository-side
verification; it runs on the deployment H100 host after these static and unit
gates pass.
