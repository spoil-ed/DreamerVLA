# OpenVLA-OFT Input-Token Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions prohibit subagent execution for this change.

**Goal:** Make `input_token_embedding [256,4096]` the only OpenVLA-OFT world-model observation sidecar and remove the competing `[56,4096]` observation route without removing other VLA families or the discrete action decoder's internal slots.

**Architecture:** Hydra task metadata exposes one `task.openvla_oft.input_tokens` contract. Collection and offline preprocessing emit projected current-frame vision patch tokens under `obs_embedding`; replay, world model, classifier, and cotrain consume the same tokenized shape. The actor may bridge those 256 source tokens into 56 action slots internally, but those slots are never persisted or configured as an observation sidecar.

**Tech Stack:** Python 3.11, PyTorch, Hydra/OmegaConf, HDF5/h5py, pytest, Bash launchers.

## Global Constraints

- Canonical source is `obs_hidden_source: input_token_embedding`.
- Canonical storage is `[T,256,4096]`, `token_count=256`, `token_dim=4096`, `wm_obs_dim=1048576`.
- Canonical Hydra namespace is `task.openvla_oft.input_tokens.*` and canonical task path is `task.openvla_oft.input_token_dir`.
- Keep all other VLA families and unrelated side routes.
- Keep the OpenVLA discrete decoder's internal 56 action slots; remove only 56-token WM/sidecar routes.
- Keep collected rollout role directories named `reward/` and `hidden/`.
- Do not relocate `data/processed_data` or change its symlink.
- Preserve all pre-existing staged and unstaged work. Use `apply_patch`; never reset or restore whole files from Git.
- Do not create implementation commits while the shared index contains unrelated staged paths; verify each task with focused tests instead.

---

### Task 1: Pin the Hydra contract to 256 input tokens

**Files:**
- Modify: `tests/unit_tests/test_coldstart_suite_configs.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_openvla_traj1_libero_matrix.py`
- Modify: `configs/task/_base_libero.yaml`
- Modify: `configs/task/openvla_onetraj_libero.yaml`
- Modify: `configs/task/openvla_onetraj_libero_10.yaml`
- Modify: `configs/task/openvla_onetraj_libero_object.yaml`
- Modify: `configs/task/openvla_onetraj_libero_spatial.yaml`
- Modify: `configs/task/openvla_onetraj_coldstart_libero.yaml`
- Modify: `configs/task/openvla_onetraj_coldstart_libero_10.yaml`
- Modify: `configs/task/openvla_onetraj_coldstart_libero_object.yaml`
- Modify: `configs/task/openvla_onetraj_coldstart_libero_spatial.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_noray.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml`
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`

**Interfaces:**
- Consumes: suite-specific OpenVLA-OFT checkpoint and dataset-statistics paths.
- Produces: `task.openvla_oft.input_tokens` with source, shape, conditioning, and action-bridge metadata used by every downstream mainline config.

- [ ] **Step 1: Write failing task-contract assertions**

Add a shared assertion to `tests/unit_tests/test_coldstart_suite_configs.py` and apply it to goal/object/spatial/10:

```python
def _assert_input_token_contract(cfg: DictConfig) -> None:
    oft = cfg.task.openvla_oft
    assert "hidden_token" not in oft
    assert oft.input_tokens.expected_obs_hidden_source == "input_token_embedding"
    assert int(oft.input_tokens.num_images_in_input) == 1
    assert int(oft.input_tokens.patches_per_image) == 256
    assert int(oft.input_tokens.token_count) == 256
    assert int(oft.input_tokens.token_dim) == 4096
    assert int(oft.input_tokens.wm_obs_dim) == 1_048_576
    assert str(oft.input_token_dir).endswith(
        "_oft_input_token_embedding_vla_policy_h1"
    )
```

Extend launcher tests to assert generated collect commands override
`task.openvla_oft.input_token_dir`, and that resolved cotrain configs use
`task.openvla_oft.input_tokens.*` rather than `hidden_token.*`.

- [ ] **Step 2: Run the contract tests and observe failure**

Run:

```bash
pytest -q \
  tests/unit_tests/test_coldstart_suite_configs.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py
```

Expected: failures show the current `hidden_token` namespace/source and 56-token dimensions.

- [ ] **Step 3: Restore one canonical task metadata block**

Make the effective task contract resolve to:

```yaml
openvla_oft:
  input_token_dir: ${task.hdf5_dir}_oft_input_token_embedding_vla_policy_h1
  input_tokens:
    expected_action_head_type: oft_discrete_token
    expected_obs_hidden_source: input_token_embedding
    expected_prompt_style: vla_policy
    expected_history: 1
    expected_include_state: false
    expected_rotate_images_180: true
    num_images_in_input: 1
    patches_per_image: 256
    token_count: 256
    token_dim: 4096
    wm_obs_dim: 1048576
    chunk_size: 8
    hidden_state_dim: 4096
    latent_stage: query_before
    latent_source: OpenVLA-OFT projected vision input-token embeddings [256,4096]
    proprio_keys: [ee_pos, ee_ori, gripper_states]
    proprio_dim: 8
    proprio_emb_dim: 10
    num_proprio_repeat: 1
    lang_dim: 4096
    lang_emb_dim: 32
    num_lang_repeat: 1
    classifier_latent_dim: 4138
    action_emb_dim: 10
    num_action_repeat: 1
    model_dim: 4148
```

Remove OpenVLA observation-sidecar fields whose only meaning is the 56-token route. Update every sync/Ray/launcher interpolation to `input_tokens.*`.

- [ ] **Step 4: Re-run the contract tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 2: Emit input-token metadata and reject semantic aliases

**Files:**
- Modify: `tests/unit_tests/test_openvla_oft_input_token_shape.py`
- Modify: `tests/unit_tests/test_collect_rollouts_config.py`
- Modify: `tests/unit_tests/test_collect_rollouts_runner.py`
- Modify: `tests/unit_tests/test_oft_rollout_bundle.py`
- Modify: `dreamervla/runners/oft_collect_common.py`
- Modify: `dreamervla/runners/rollout_hidden_extractor.py`
- Modify: `dreamervla/runners/collect_rollouts_runner.py`
- Modify: `dreamervla/runners/collect_parallel_rollouts.py`
- Modify: `dreamervla/workers/inference/oft_rollout.py`

**Interfaces:**
- Consumes: `collect.oft_latent_spec` resolved from Task 1.
- Produces: tokenized `obs_embedding` arrays and `preprocess_config.json` with an unambiguous `input_token_embedding` source.

- [ ] **Step 1: Restore failing input-token collector tests**

Make `test_input_token_preprocess_config_records_dim_decomposition` assert:

```python
assert out["obs_hidden_source"] == "input_token_embedding"
assert out["token_count"] == 256
assert out["token_dim"] == 4096
assert out["hidden_dim"] == 256 * 4096
assert out["obs_embedding_shape"] == [256, 4096]
assert out["hidden_storage_format"] == "tokenized"
```

Add a test that `make_preprocess_config` rejects `expected_obs_hidden_source="hidden_token"` with a message directing callers to `input_token_embedding`.

- [ ] **Step 2: Run the collector tests and observe failure**

Run:

```bash
pytest -q \
  tests/unit_tests/test_openvla_oft_input_token_shape.py \
  tests/unit_tests/test_collect_rollouts_config.py \
  tests/unit_tests/test_collect_rollouts_runner.py \
  tests/unit_tests/test_oft_rollout_bundle.py
```

Expected: source-name and branch-selection failures.

- [ ] **Step 3: Restore projected input-token extraction**

Expose the helper with this role-specific interface:

```python
def input_token_embedding_from_projected(
    projected: torch.Tensor,
    *,
    image_keys: Sequence[str],
    patches_per_image: int,
) -> torch.Tensor:
    token_count = len(tuple(image_keys)) * int(patches_per_image)
    if projected.ndim != 3 or projected.shape[1] < token_count:
        raise ValueError(
            "projected vision tokens cannot satisfy input-token contract: "
            f"shape={tuple(projected.shape)} token_count={token_count}"
        )
    return projected[:, -token_count:, :]
```

Make the OpenVLA collector accept only `input_token_embedding` as its observation source, preserve action hidden states only for action decoding, and emit the projected token tensor as `[N,4096]`. Rename `_hidden_token_sidecar_dims` and `vla_latent_spec` documentation to input-token terminology. Keep generic `hidden_dir` transport names unchanged.

- [ ] **Step 4: Re-run the collector tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 3: Validate collection schema before resume or warmup

**Files:**
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`

**Interfaces:**
- Consumes: collected `reward/`, `hidden/`, `preprocess_config.json`, and representative `obs_embedding` datasets.
- Produces: `validate_collected_outputs(...) -> list[str]` errors that prevent stale 56-token collections from being counted as complete.

- [ ] **Step 1: Add failing schema-validation tests**

Add fixtures for a valid `[T,256,4096]` sidecar and an invalid `[T,56,4096]` sidecar. Assert:

```python
errors = validate_collected_outputs(reward_dir=reward, hidden_dir=hidden)
assert errors == []

errors = validate_collected_outputs(reward_dir=reward56, hidden_dir=hidden56)
assert any("obs_hidden_source" in item for item in errors)
assert any("token_count" in item for item in errors)
assert any("obs_embedding shape" in item for item in errors)
```

Add a `collect_resume` test proving a count-complete 56-token collection raises before the "target already collected" branch.

- [ ] **Step 2: Run the launcher tests and observe failure**

Run:

```bash
pytest -q tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py
```

Expected: the existing validator accepts the incompatible fixture.

- [ ] **Step 3: Implement strict input-token validation**

Validate JSON metadata and the first available `data/demo_*/obs_embedding` dataset against:

```python
EXPECTED_OPENVLA_INPUT_TOKEN_SCHEMA = {
    "obs_hidden_source": "input_token_embedding",
    "token_count": 256,
    "token_dim": 4096,
    "hidden_storage_format": "tokenized",
}
```

Require a rank-3 episode dataset whose trailing dimensions are `(256, 4096)`. Call this validator before `summarize_collection` can skip collection and before `skip_collect` can pass data to cotrain. Return precise errors; do not automatically rename or delete data.

- [ ] **Step 4: Re-run the launcher tests**

Run the command from Step 2.

Expected: all launcher tests pass and the invalid fixture is rejected.

### Task 4: Converge offline preprocessing on input-token sidecars

**Files:**
- Rename: `dreamervla/preprocess/preprocess_oft_hidden_token.py` -> `dreamervla/preprocess/preprocess_oft_input_tokens.py`
- Rename: `configs/scripts/preprocess_oft_hidden_token.yaml` -> `configs/scripts/preprocess_oft_input_tokens.yaml`
- Rename: `scripts/preprocess/35_oft_hidden_token.sh` -> `scripts/preprocess/35_oft_input_tokens.sh`
- Modify: `configs/scripts/preprocess_suite.yaml`
- Modify: `tests/unit_tests/test_preprocess_oft_policy_mode.py`
- Modify: `tests/unit_tests/test_preprocess_language_sidecar.py`
- Modify: `tests/unit_tests/test_preprocess_check_artifacts.py`
- Modify: `tests/unit_tests/test_hydra_script_config.py`

**Interfaces:**
- Consumes: reward HDF5, OpenVLA-OFT one-trajectory checkpoint, one camera, history one.
- Produces: only `_oft_input_token_embedding_vla_policy_h1` HDF5 sidecars plus `preprocess_config.json`.

- [ ] **Step 1: Add failing preprocessing surface tests**

Assert the Hydra config and shell script expose only:

```yaml
out_input_token_dir: ..._oft_input_token_embedding_vla_policy_h1
obs_hidden_source: input_token_embedding
```

Assert removed arguments such as `out_c_dir`, `out_d_dir`, `out_hidden_token_flat_dir`, and `out_hidden_token_dir` are absent from the public config.

- [ ] **Step 2: Run preprocessing tests and observe failure**

Run:

```bash
pytest -q \
  tests/unit_tests/test_preprocess_oft_policy_mode.py \
  tests/unit_tests/test_preprocess_language_sidecar.py \
  tests/unit_tests/test_preprocess_check_artifacts.py \
  tests/unit_tests/test_hydra_script_config.py
```

Expected: old filenames, output arguments, and `hidden_token` metadata fail assertions.

- [ ] **Step 3: Rename and simplify the preprocessing route**

Pin policy mode to the discrete one-trajectory checkpoint and reject L1 component checkpoints. Keep image preparation, projected vision-token extraction, language embedding, HDF5 atomic writes, distributed sharding, completion attributes, and artifact validation. Delete C/D MLP sidecar output and the OpenVLA action-query/56-token sidecar output. The only episode dataset written by this module is:

```python
demo_out.create_dataset(
    "obs_embedding",
    shape=(length, token_count, token_dim),
    dtype=np.dtype(args.output_dtype),
    chunks=(1, token_count, token_dim),
)
```

Write `obs_hidden_source="input_token_embedding"`, `token_count=256`, `token_dim=4096`, and `hidden_storage_format="tokenized"` into both file attributes and `preprocess_config.json`.

- [ ] **Step 4: Re-run preprocessing tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 5: Remove OpenVLA 56-token observation-route references

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `configs/README.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/reference/model_datasets/openvla_oft_libero_goal.md`
- Modify: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Modify: `docs/tutorials/experiments/EXPLAINED.md`
- Modify: `spec/06_routes.md`
- Modify: `configs/experiment/latent_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `configs/experiment/wm_full_dataset_train.yaml`
- Modify: `dreamervla/config.py`
- Modify: `dreamervla/diagnostics/benchmark_manual_workers.py`
- Modify: `dreamervla/diagnostics/diagnose_ppo_imagine_vs_real.py`
- Modify: `dreamervla/diagnostics/experiment_stage_checks.py`
- Modify: `dreamervla/diagnostics/wm_single_trajectory_overfit.py`
- Modify: `dreamervla/runners/embodied_eval_runner.py`
- Modify: `scripts/collect_parallel.sh`
- Modify: `tests/e2e_tests/test_s6_ray_real_oft_collect.py`
- Modify: `tests/e2e_tests/test_s6_real_oft_coldstart.py`
- Modify: `tests/unit_tests/test_config_validation.py`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`
- Modify: `tests/unit_tests/test_ray_coldstart_real_config.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`
- Modify: `tests/unit_tests/test_spec_docs.py`

**Interfaces:**
- Consumes: final config/code names from Tasks 1-4.
- Produces: repository documentation and hygiene checks that distinguish 256-token observation state from internal action slots.

- [ ] **Step 1: Add a failing repository hygiene gate**

Add assertions that OpenVLA mainline files contain none of:

```python
FORBIDDEN_OPENVLA_WM_PATTERNS = (
    "task.openvla_oft.hidden_token",
    "expected_obs_hidden_source: hidden_token",
    "OpenVLA-OFT discrete hidden_token [56,4096]",
    "wm_obs_dim: 229376",
)
```

Scope the search to OpenVLA mainline config, runner, preprocessing, launcher, docs, and tests; exclude other VLA implementations and explicit action-slot code.

- [ ] **Step 2: Run hygiene tests and observe failure**

Run:

```bash
pytest -q \
  tests/unit_tests/test_repository_hygiene.py \
  tests/unit_tests/test_spec_docs.py
```

Expected: current mainline files still contain forbidden 56-route references.

- [ ] **Step 3: Update the listed mainline references**

Run this audit repeatedly while editing:

```bash
rg -n \
  'task\.openvla_oft\.hidden_token|expected_obs_hidden_source: hidden_token|wm_obs_dim: 229376|\[56, ?4096\]' \
  AGENTS.md README.md README.zh-CN.md configs dreamervla scripts docs spec tests
```

Within the files listed for this task, apply these exact rules:

```text
task.openvla_oft.hidden_token.*          -> task.openvla_oft.input_tokens.*
obs_hidden_source=hidden_token           -> obs_hidden_source=input_token_embedding
OpenVLA WM token_count=56                -> token_count=256
OpenVLA WM wm_obs_dim=229376             -> wm_obs_dim=1048576
*_oft_hidden_token_vla_policy_h1         -> *_oft_input_token_embedding_vla_policy_h1
```

Delete a file only when its entire public purpose is an OpenVLA 56-token
observation-sidecar route. Do not apply these replacements inside RynnVLA files
or inside `OpenVLADiscreteTokenActor`/`LatentToOpenVLAHiddenStateActor` action-slot
implementation and tests.

- [ ] **Step 4: Re-run hygiene tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 6: Remove only local OpenVLA 56-token sidecar artifacts

**Files:**
- Delete: only runtime directories positively identified by metadata as OpenVLA `action_query`/56-token observation sidecars.
- Preserve: all `input_token_embedding [T,256,4096]` directories and all other-VLA directories.

**Interfaces:**
- Consumes: `preprocess_config.json` and representative HDF5 dataset shapes.
- Produces: an audit log in command output and a runtime tree without the removed OpenVLA 56-token observation route.

- [ ] **Step 1: Inventory candidates without deletion**

Run a Python scanner over every `data/**/preprocess_config.json`. For each config, print its directory, `obs_hidden_source`, `token_count`, `token_dim`, and the first `obs_embedding` trailing shape. Classify as delete only when all available evidence identifies OpenVLA and the observation route is `action_query` with 56 tokens or `hidden_token` with 56 tokens.

- [ ] **Step 2: Review the generated candidate list against preservation rules**

Assert the list excludes:

```text
data/datasets/processed_data/libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h1
data/collected_rollouts/libero_goal
data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal/online_cotrain_backbone_latent
```

when their metadata/HDF5 shape is `input_token_embedding [256,4096]`. Exclude every RynnVLA directory regardless of its token dimensions.

- [ ] **Step 3: Delete confirmed candidate directories and their paired reward shards**

Delete only the reviewed candidate directories. When a candidate is a collected `hidden/` directory paired with a `reward/` directory for the same obsolete collection, delete the pair together so no orphan can be counted by resume logic.

- [ ] **Step 4: Re-run the scanner**

Expected: zero OpenVLA 56-token observation-sidecar candidates; preserved input-token and other-VLA directories remain.

### Task 7: Full verification

**Files:**
- Verify only; do not modify unless a failing check identifies a scoped regression.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: fresh evidence for the final handoff.

- [ ] **Step 1: Run focused mainline tests**

```bash
pytest -q \
  tests/unit_tests/test_openvla_oft_input_token_shape.py \
  tests/unit_tests/test_coldstart_suite_configs.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py \
  tests/unit_tests/test_collect_rollouts_config.py \
  tests/unit_tests/test_collect_rollouts_runner.py \
  tests/unit_tests/test_oft_rollout_bundle.py \
  tests/unit_tests/test_online_cotrain_pipeline.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_preprocess_oft_policy_mode.py \
  tests/unit_tests/test_repository_hygiene.py \
  tests/unit_tests/test_spec_docs.py
```

Expected: zero failures.

- [ ] **Step 2: Compose all four suite routes**

Compose `collect_rollouts_onetraj`, `collect_rollouts_ray`, `openvla_onetraj_libero_cotrain_noray`, and `openvla_onetraj_libero_cotrain_ray` for goal/object/spatial/10. Assert every resolved WM/classifier source has `token_count=256`, `token_dim=4096`, `wm_obs_dim=1048576`, and `obs_hidden_source=input_token_embedding`.

- [ ] **Step 3: Run source and formatting checks**

```bash
git diff --check
python -m compileall -q dreamervla
```

Expected: both commands exit 0.

- [ ] **Step 4: Verify the dirty-worktree boundary**

Inspect `git status --short` and `git diff --stat`. Confirm no unrelated pre-existing file was reverted and no runtime input-token or other-VLA artifact was deleted.
