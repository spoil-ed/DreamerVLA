# OpenVLA-OFT Input-Token Mainline Convergence

## Objective

The only OpenVLA-OFT world-model observation latent on the DreamerVLA mainline is
the current-frame projected vision input-token embedding:

```text
obs_hidden_source: input_token_embedding
obs_embedding: [T, 256, 4096]
token_count: 256
token_dim: 4096
wm_obs_dim: 1048576
history: 1
num_images_in_input: 1
patches_per_image: 256
```

The repository must not expose the OpenVLA action-query/hidden-token tensor
`[T, 56, 4096]` as an alternative world-model observation sidecar or cotrain
state.

## Canonical Naming

- Hydra metadata namespace: `task.openvla_oft.input_tokens.*`.
- Task path field: `task.openvla_oft.input_token_dir`.
- Sidecar metadata value: `obs_hidden_source: input_token_embedding`.
- Sidecar directory suffix: `_oft_input_token_embedding_vla_policy_h1`.
- HDF5 dataset key: `obs_embedding`.
- Collected rollout role directory: `hidden/`; this role-based channel name may
  remain because replay APIs pair `reward/` and `hidden/` independently of the
  concrete latent source.

Names such as `hidden_token` must not be used as aliases for projected vision
input tokens. Generic references to a hidden-sidecar channel may remain when
they do not claim a concrete latent source.

## Mainline Data Flow

```text
OpenVLA-OFT image preprocessing
  -> projected current-frame vision patch tokens [256,4096]
  -> collected_rollouts/<suite>/hidden/*.hdf5:obs_embedding
  -> OnlineReplay
  -> world model + classifier warmup
  -> sync or manual-Ray online cotrain
```

The world model and classifier consume 256 source tokens. The actor bridge may
still produce 56 OpenVLA action slots internally before the LM head decodes the
action chunk. Those action slots are an implementation detail of action
decoding, not a world-model observation latent, and must not be removed.

## Code and Configuration Changes

1. Restore every mainline collector, warmup, sync-cotrain, and Ray-cotrain
   reference from `task.openvla_oft.hidden_token.*` to
   `task.openvla_oft.input_tokens.*`.
2. Make all four supported LIBERO task configs (`goal`, `object`, `spatial`, and
   `10`) declare the same one-image, history-one, 256-token input-token
   contract, with suite-specific checkpoints and statistics unchanged.
3. Remove the OpenVLA `[56,4096]` observation-sidecar fields, configs,
   experiments, extraction branches, validators, tests, and documentation.
4. Rename the OpenVLA offline preprocessing entry point to the input-token role
   and make it emit only `input_token_embedding` sidecars. Remove its C/D and
   action-query/56-token sidecar outputs.
5. Rename projected-token helpers and metadata from ambiguous `hidden_token`
   wording to `input_token_embedding` wording.
6. Keep shared world-model, replay, classifier, actor, and rollout abstractions
   when the 256-token mainline still imports them.
7. Keep other VLA families and unrelated side routes. Their own latent formats
   must not be renamed to OpenVLA input tokens.
8. Keep the internal 56-slot OpenVLA discrete action decoder and its tests,
   while naming it explicitly as action slots rather than a WM sidecar.

## Local Data Policy

- Preserve sidecars whose metadata and HDF5 shape identify
  `input_token_embedding [T,256,4096]`.
- Preserve data belonging to other VLA families.
- Delete only OpenVLA world-model/sidecar artifacts whose metadata identifies
  the removed 56-token observation route (`action_query` or a 56-token
  `hidden_token` source).
- Do not classify an artifact from its directory name alone. Inspect
  `preprocess_config.json` and representative HDF5 shapes first.
- Do not relocate `data/processed_data` or change its symlink in this change.

## Compatibility and Failure Handling

Cold-start reuse must validate the stored sidecar contract before counting an
existing collection as complete. At minimum it must reject a mismatch in:

- `obs_hidden_source`;
- `token_count`;
- `token_dim`;
- `hidden_storage_format`;
- representative `obs_embedding` shape.

An incompatible collection must be reported explicitly and must not be silently
renamed, counted as complete, or passed into warmup.

## Verification

The change is complete only when all of the following hold:

1. Hydra composition succeeds for sync and Ray mainline experiments across all
   four LIBERO suites.
2. Resolved world-model and classifier configs use `token_count=256`,
   `token_dim=4096`, and `wm_obs_dim=1048576`.
3. Collector and offline-preprocessor tests emit
   `obs_hidden_source=input_token_embedding` and `[T,256,4096]` storage.
4. A synthetic 56-token OpenVLA sidecar is rejected before collection resume or
   warmup.
5. Repository hygiene checks find no OpenVLA WM/sidecar configuration that uses
   56 tokens or exposes `task.openvla_oft.hidden_token.*`.
6. Tests for the internal 56-slot action decoder continue to pass.
7. The focused unit suite and repository hygiene suite pass without relying on
   deleted compatibility aliases.

## Non-Goals

- Removing other VLA families or unrelated experimental routes.
- Removing the OpenVLA discrete action decoder's internal action slots.
- Migrating all runtime data into a new root.
- Rebuilding expensive input-token sidecars that already satisfy the canonical
  schema.
