# DinoWM Query-Stage World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DreamerVLA world models explicitly split into query-before and query-after routes, while the chunk world model follows the DINO-WM concat-conditioning pattern and keeps all architecture/parameter decisions in Hydra.

**Architecture:** The world model consumes precomputed VLA hidden sidecars as token sequences. `latent_stage=query_after` means action-query/action-hidden sidecars; `latent_stage=query_before` means backbone/input-token sidecars before action query. `ChunkAwareDinoWMWorldModel` uses DINO-WM `concat_dim=1`: action is encoded and concatenated to every observation token channel, not added as a separate token.

**Tech Stack:** Hydra configs, PyTorch `nn.Module`, DreamerVLA runner/world-model contracts, pytest.

---

## Non-Negotiable Constraints

- Model and dataset remain decoupled. Dataset/task configs define sidecar shape and query stage; model configs consume those fields.
- Hydra is the source of truth for architecture parameters. Code may keep constructor defaults for legacy/manual tests only, but routes must set `model_dim`, `action_emb_dim`, `depth`, `heads`, `dim_head`, and `mlp_dim` explicitly.
- YAML must not rely on arithmetic. Computed relationships are written as explicit values and validated in `dreamervla/config.py`.
- No multi-node scope.
- Optional components are detected by existence of configured fields, not by hardcoded blockers.

## Current Classification

| Route family | `latent_stage` | Sidecar meaning | Shape examples |
|---|---|---|---|
| Rynn action hidden | `query_after` | action-query/action-hidden latent after VLA action query | LIBERO goal/object `35 x 1024`; spatial/10 `70 x 1024` |
| Rynn input tokens | `query_before` | current-frame Chameleon/VLA input-token latent before action query | `2048 x 4096` |
| OpenVLA-OFT action hidden | `query_after` | OFT action-query hidden after action query | `56 x 4096` |
| OpenVLA-OFT input tokens | `query_before` | projected vision/backbone tokens before action query | `512 x 4096` |

## DinoWM-Style Conditioning Contract

For one time step:

```text
obs tokens:       [N, token_dim]
raw action:       [7]
action encoder:   [7] -> [action_emb_dim]
tile action:      [N, action_emb_dim]
concat token:     [N, token_dim + action_emb_dim * num_action_repeat]
```

The model predicts only the observation part:

```text
z_pred:    [B, H, N, model_dim]
pred_obs:  z_pred[..., :token_dim]
next e:    pred_obs[:, -1]  # [B, N, token_dim]
```

The predicted action-condition channel is discarded. The next rollout step uses the externally supplied next action from dataset, actor, or planner.

## Transformer Sizing Policy

The residual width is fixed by the sidecar token space:

```text
model_dim = token_dim + action_emb_dim * num_action_repeat
```

Target explicit Hydra values (OpenVLA discrete routes; Rynn rows unchanged). The
query_after rows are already applied; the query_before row lands in Task 8:

| Token family | `token_dim` | `action_emb_dim` | `model_dim` | `depth` | `heads` | `dim_head` | `mlp_dim` | inner = `heads*dim_head` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rynn query-after | 1024 | 10 | 1034 | 6 | 16 | 64 | 2048 | 1024 |
| Rynn query-before | 4096 | 10 | 4106 | 4 | 8 | 32 | 1024 | 256 |
| OpenVLA-OFT query-after (action-hidden) | 4096 | 10 | 4106 | 6 | 16 | 256 | 4096 | 4096 (1.00x) |
| OpenVLA-OFT query-before (input-token) | 4096 | 10 | 4106 | 6 | 16 | 128 | 2048 | 2048 (0.50x) |

Sizing rationale (efficiency vs capacity, 1B hard cap, discrete-only):

- `model_dim` is pinned by DINO-WM concat: `token_dim + action_emb_dim * num_action_repeat`
  (= 4106 for OpenVLA). Only the predictor internal width (`heads*dim_head`,
  `mlp_dim`, `depth`) is free. DINO-WM reference uses `action_emb_dim=10`, `concat_dim=1`.
- Allocate capacity by sequence length. query_after carries only
  `num_hist*token_count = 3*56 = 168` tokens, so full-width attention is nearly
  free -> spend there: `inner = 4096 = model_dim` (no compression of the dense
  4096-d VLA tokens) with a lean `mlp_dim = model_dim`. ~610M.
- query_before carries `3*512 = 1536` tokens (~9x costlier per unit of capacity),
  so stay lean: half-width attention `inner = 2048` (still 8x the old compact 256,
  de-bottlenecked) with `mlp_dim = 2048`. ~313M.
- The old compact profile (`inner=256, mlp=1024`) under-parameterized the 4096-d
  tokens (attention saw 6% of the residual width) and is superseded for OpenVLA.
- Positional embedding is sized for the active DINO-WM window `num_hist * token_count`,
  not `max_seq_len * token_count`.

Measured total parameters (via `hydra.utils.instantiate`, the runner path):

| OpenVLA route | seq `num_hist*token_count` | inner | `mlp_dim` | `depth` | total params |
|---|---:|---:|---:|---:|---:|
| query-after (action-hidden) | 168 | 4096 | 4096 | 6 | 610.5M |
| query-before (input-token) | 1536 | 2048 | 2048 | 6 | 313.4M |

Reference points for query-after: compact `inner=256,mlp=1024,depth=4` = 55.5M;
dino-wm default `inner=1024,mlp=2048,depth=6` = 206.9M; full-width
`inner=4096,mlp=8192,depth=6` = 812.4M (all under 1B).

## Task 1: Keep Query-Stage Metadata Explicit

**Files:**
- Modify: `configs/task/libero_goal.yaml`
- Modify: `configs/task/libero_object.yaml`
- Modify: `configs/task/libero_spatial.yaml`
- Modify: `configs/task/libero_10.yaml`
- Modify: `configs/task/OpenVLA_Onetraj_LIBERO*.yaml`
- Modify: `configs/worldmodel/*.yaml`
- Modify: `configs/dreamervla/*.yaml`
- Modify: `configs/experiment/online_cotrain_oft_backbone_latent.yaml`
- Test: `tests/unit_tests/test_config_validation.py`

- [x] Add `latent_stage: query_after` to action-query/action-hidden task specs.
- [x] Add `latent_stage: query_before` to input-token/backbone task specs.
- [x] Propagate `world_model.latent_stage` from the selected task spec.
- [x] Add validator checks for allowed values and world-model/task sidecar agreement.
- [x] Test task specs expose the expected stages.

Verification:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_config_validation.py -q
```

Expected: all config validation tests pass.

## Task 2: Use DINO-WM Concat Conditioning

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py`
- Modify: `dreamervla/models/world_model/dino_wm.py`
- Test: `tests/unit_tests/test_chunk_wm_autoregressive.py`

- [x] Replace the single action-token path with action-channel concat.
- [x] Add DINO-WM-style attention where residual `model_dim` is independent of `heads * dim_head`.
- [x] Keep action source external: dataset actions for WM training; actor/planner actions for rollout.
- [x] Ensure `predict_next_chunk` remains autoregressive over `action_chunk`.
- [x] Add tests proving action is tiled to every obs token without increasing token count.

Verification:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_chunk_wm_autoregressive.py -q
```

Expected: chunk WM tests pass.

## Task 3: Put Transformer Sizing in Hydra

**Files:**
- Modify: `configs/worldmodel/rynnvla_action_chunk.yaml`
- Modify: `configs/worldmodel/rynnvla_input_token_chunk.yaml`
- Modify: `configs/worldmodel/openvla_oft_action_chunk.yaml`
- Modify: `configs/worldmodel/openvla_oft_input_token_chunk.yaml`
- Modify: `configs/dreamervla/rynnvla_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/openvla_oft_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/*input_token*_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/online_cotrain_pipeline_openvla_oft_action_hidden.yaml`
- Modify: `configs/experiment/online_cotrain_oft_backbone_latent.yaml`
- Modify: `dreamervla/config.py`
- Test: `tests/unit_tests/test_config_validation.py`

- [x] Set `action_emb_dim`, `num_action_repeat`, and explicit `model_dim` in Hydra.
- [x] Set `depth`, `heads`, `dim_head`, and `mlp_dim` explicitly in Hydra.
- [x] Use compact transformer budget for 4096-d token routes.
- [x] Validate `model_dim == token_dim + action_emb_dim * num_action_repeat`.
- [x] Validate chunk WM transformer sizing keys are present in Hydra.

Verification:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_config_validation.py::test_query_before_world_model_routes_use_compact_transformer_budget -q
```

Expected: query-before routes resolve to `model_dim=4106`, `depth=4`, `heads=8`, `dim_head=32`, `mlp_dim=1024`.

## Task 4: Fix Positional Embedding Scaling

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py`
- Test: `tests/unit_tests/test_chunk_wm_autoregressive.py`

- [x] Size `pos_embedding` as `[1, num_hist * token_count, model_dim]`.
- [x] Reject calls to `encode` with more frames than the configured DINO-WM context.
- [x] Preserve `max_seq_len` as external sequence/dataset validation metadata, not the DINO-WM predictor position-table length.

Verification:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_chunk_wm_autoregressive.py::test_chunk_wm_position_embedding_scales_with_history_not_max_seq_len -q
```

Expected: `pos_embedding.shape == (1, num_hist * token_count, model_dim)`.

## Task 5: Recompute and Record Parameter Counts

**Files:**
- Create or update: `docs/superpowers/plans/2026-06-19-dinowm-query-stage-worldmodel.md`

- [x] Instantiate representative Hydra routes.
- [x] Count total, predictor, action projection, reward head, and position-embedding parameters.
- [x] Record current parameter budget table in this document.
- [ ] Re-run this count after any Hydra transformer budget change.

Command:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.models.world_model.dino_wm_chunk import ChunkAwareDinoWMWorldModel

keys = {
    "obs_dim", "action_dim", "token_count", "token_dim", "time_horizon",
    "model_dim", "depth", "heads", "dim_head", "mlp_dim", "dropout",
    "num_hist", "num_pred", "max_seq_len", "hidden_loss_scale",
    "cosine_loss_scale", "reward_head_type", "reward_loss_scale",
    "reward_hidden_dim", "reward_init_logit", "reward_pos_weight",
    "return_predictions", "chunk_size", "chunk_rollout_chunks",
    "chunk_rollout_loss_scale", "action_emb_dim", "num_action_repeat",
    "latent_stage", "latent_source", "freeze_backbone",
}

routes = [
    ("rynn_query_after", ["experiment=world_model_dinowm_chunk", "task=libero_goal"]),
    ("rynn_query_before", ["experiment=world_model_dinowm_chunk", "task=libero_goal", "worldmodel=rynnvla_input_token_chunk"]),
    ("oft_query_after", ["experiment=oft_world_model_dinowm_chunk", "task=OpenVLA_Onetraj_LIBERO"]),
    ("oft_query_before", ["experiment=oft_world_model_dinowm_chunk", "task=OpenVLA_Onetraj_LIBERO", "worldmodel=openvla_oft_input_token_chunk"]),
]

with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    for name, overrides in routes:
        cfg = compose(config_name="train", overrides=overrides)
        wm_cfg = OmegaConf.to_container(cfg.world_model, resolve=True)
        wm_cfg.pop("_target_", None)
        wm_cfg = {k: v for k, v in wm_cfg.items() if k in keys}
        model = ChunkAwareDinoWMWorldModel(**wm_cfg)
        total = sum(p.numel() for p in model.parameters())
        predictor = sum(p.numel() for p in model.predictor.parameters())
        print(name, wm_cfg["latent_stage"], wm_cfg["token_count"], wm_cfg["token_dim"], wm_cfg["model_dim"], total, predictor, model.pos_embedding.numel())
PY
```

Expected: counts stay in the same order of magnitude as the table above.

## Task 6: Verification Matrix

**Files:**
- Test: `tests/unit_tests/test_chunk_wm_autoregressive.py`
- Test: `tests/unit_tests/test_eval_chunkwm_closeloop.py`
- Test: `tests/unit_tests/test_config_validation.py`
- Test: `tests/unit_tests/test_runner_public_api.py`
- Test: `tests/unit_tests/test_online_cotrain_ray_runner.py`
- Test: `tests/e2e_tests/test_s6_ray_coldstart_collect.py`

- [x] Run focused config/model tests after query-stage and concat changes.
- [x] Run Ray runner public API tests after Hydra field additions.
- [x] Run Ray coldstart synthetic smoke after config validation changes.
- [ ] Re-run the full focused matrix after the final parameter-budget edits.
- [ ] Run `git diff --check`.
- [ ] Run `py_compile` for modified Python files.

Focused command:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_chunk_wm_autoregressive.py \
  tests/unit_tests/test_eval_chunkwm_closeloop.py \
  tests/unit_tests/test_config_validation.py \
  tests/unit_tests/test_runner_public_api.py \
  tests/unit_tests/test_online_cotrain_ray_runner.py \
  -q
```

Expected: all selected tests pass.

Ray smoke command:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q
```

Expected: synthetic Ray coldstart tests pass.

Static checks:

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m py_compile \
  dreamervla/models/world_model/dino_wm_chunk.py \
  dreamervla/models/world_model/dino_wm.py \
  dreamervla/config.py \
  dreamervla/diagnostics/eval_chunkwm_closeloop.py

git diff --check
```

Expected: exit code 0.

## Task 7: New Training Smoke After Architecture Change

**Files:**
- Use: `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`
- Use configs under: `configs/dreamervla/online_cotrain_pipeline_openvla_oft_action_hidden.yaml`

- [ ] Do not reuse old `model_dim=1024` warmup checkpoints for the new concat architecture.
- [ ] Run a reduced OpenVLA query-after warmup smoke with the compact 4106-d transformer profile.
- [ ] Record whether memory fits at the current batch/env settings.
- [ ] If memory does not fit, change only Hydra throughput knobs: batch size, env count, gradient accumulation, sequence length.
- [ ] Run closed-loop latent diagnostic on the new checkpoint only.
- [ ] Update the tutorial with the new architecture and incompatibility note.

Expected result:

- WM loss runs without shape errors.
- Checkpoint `resolved_config.yaml` shows `latent_stage=query_after`, `model_dim=4106`, compact transformer params from Hydra.
- Closed-loop diagnostic is reported separately from the old 1024-projection result.

## Current Status Snapshot

- Implemented: query-stage metadata, DINO-WM concat action conditioning, Hydra validation, compact 4096-d profile, position embedding sized to the active history window.
- Verified before the final parameter-budget edits:
  - 53 focused unit/API tests passed.
  - Ray coldstart collect synthetic e2e passed: 4 tests.
  - `py_compile` and `git diff --check` passed.
- Verified after adding compact query-before/profile tests:
  - `tests/unit_tests/test_chunk_wm_autoregressive.py`
  - `tests/unit_tests/test_config_validation.py`
  - `tests/unit_tests/test_online_cotrain_ray_runner.py`
  - result: 34 tests passed.
- Remaining immediate work: rerun the full focused matrix and static checks after the latest parameter-budget edits, then update the OpenVLA tutorial.

---

# Upper-Bound Probe Pipeline (full discrete OpenVLA -> existing-data latents -> WM ceiling)

**Goal:** Measure the *ceiling* of the DINO-WM chunk world model by training it on
latents from a FULLY-trained discrete OpenVLA-OFT (not the 1-trajectory cold-start
checkpoint), extracted from EXISTING LIBERO demos with NO environment rollouts,
for both latent schemes (query_after action-hidden, query_before input-token).

**Why this differs from cotrain:** the cold-start cotrain tutorial uses the weak
`Openvla-oft-SFT-traj1` policy; its latents cap the WM ceiling artificially. Here
we SFT a full-data discrete OpenVLA-OFT first, so the latents are as informative as
the VLA can make them.

**Scope:** OpenVLA-OFT discrete only (`use_l1_regression=false`, no L1 head); start
on `libero_goal`; WM predictor under the 1B cap.

**Pipeline:**

```text
[Task 8]  re-size query_before WM (313M) + update its pinning test    (only code/test change)
[Task 9]  SFT full discrete OpenVLA-OFT on libero_goal (train_vla)     -> full-data discrete ckpt
[Task 10] build reward HDF5 from existing demos, extract BOTH sidecars (no rollout)
[Task 11] train query_after WM (610M) on the sidecars                  -> next-latent loss
[Task 12] train query_before WM (313M) on the sidecars                 -> next-latent loss
[Task 13] closed-loop latent diagnostic on both                        -> the ceiling curve
[Task 14] (optional) scale to object / spatial / 10
```

All stages assume `export DVLA_DATA_ROOT="$(pwd -P)/data"` and the OFT transformers
fork verified via `bash scripts/install/60_verify.sh`.

## Task 8: Re-size the query_before WM and update its pinning test

The query_after route was already moved to the balanced 610M profile
(`configs/worldmodel/openvla_oft_action_chunk.yaml`,
`configs/dreamervla/online_cotrain_pipeline_openvla_oft_action_hidden.yaml`). The
query_before route is still on the old compact budget and is pinned by a test, so
re-sizing it is the only code/test change in this pipeline.

**Files:**
- Modify: `configs/worldmodel/openvla_oft_input_token_chunk.yaml`
- Modify: `tests/unit_tests/test_config_validation.py`

- [ ] **Step 1: Update the pinning test to expect the lean-debottlenecked OpenVLA profile.**

Replace the body of `test_query_before_world_model_routes_use_compact_transformer_budget`
(keep the name for continuity with Task 3's verification command) so the two routes
are asserted separately — Rynn stays compact, OpenVLA becomes half-width:

```python
def test_query_before_world_model_routes_use_compact_transformer_budget() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    rynn = [
        "experiment=world_model_dinowm_chunk",
        "task=libero_goal",
        "worldmodel=rynnvla_input_token_chunk",
    ]
    oft = [
        "experiment=oft_world_model_dinowm_chunk",
        "task=OpenVLA_Onetraj_LIBERO",
        "worldmodel=openvla_oft_input_token_chunk",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        rynn_cfg = compose(config_name="train", overrides=rynn)
        oft_cfg = compose(config_name="train", overrides=oft)

    # Rynn query-before stays on the compact budget.
    validate_cfg(rynn_cfg, world_size=1)
    assert rynn_cfg.world_model.latent_stage == "query_before"
    assert rynn_cfg.world_model.depth == 4
    assert rynn_cfg.world_model.heads == 8
    assert rynn_cfg.world_model.dim_head == 32
    assert rynn_cfg.world_model.mlp_dim == 1024

    # OpenVLA query-before: lean-debottlenecked half-width profile (~313M).
    validate_cfg(oft_cfg, world_size=1)
    assert oft_cfg.world_model.latent_stage == "query_before"
    assert oft_cfg.world_model.token_dim == 4096
    assert oft_cfg.world_model.model_dim == 4106
    assert oft_cfg.world_model.depth == 6
    assert oft_cfg.world_model.heads == 16
    assert oft_cfg.world_model.dim_head == 128
    assert oft_cfg.world_model.mlp_dim == 2048
```

- [ ] **Step 2: Run the test, confirm it FAILS** (config still compact).

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_config_validation.py::test_query_before_world_model_routes_use_compact_transformer_budget -q
```

Expected: FAIL on `assert oft_cfg.world_model.dim_head == 128` (still 32).

- [ ] **Step 3: Update the config** `configs/worldmodel/openvla_oft_input_token_chunk.yaml`, the `world_model` block:

```yaml
  # DINO-WM lean-debottlenecked predictor (query_before / input-token, seq=1536):
  # half-width attention inner = heads*dim_head = 2048 = 0.5*model_dim (8x the old
  # compact inner=256), with a lean FFN mlp_dim = 2048. ~313M total. The 1536-token
  # sequence is ~9x query_after's, so capacity is kept lean here on purpose.
  depth: 6
  heads: 16
  dim_head: 128
  mlp_dim: 2048
```

- [ ] **Step 4: Run the test + the full config-validation file, confirm PASS.**

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_config_validation.py -q
```

Expected: all pass.

- [ ] **Step 5: Confirm the instantiated size is ~313M and under 1B.**

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="train", overrides=[
        "experiment=oft_world_model_dinowm_chunk", "task=OpenVLA_Onetraj_LIBERO",
        "worldmodel=openvla_oft_input_token_chunk"])
    m = instantiate(cfg.world_model)
    tot = sum(p.numel() for p in m.parameters())
    print(f"query_before total = {tot/1e6:.1f}M  ({'<1B OK' if tot < 1e9 else '>1B!!'})")
PY
```

Expected: `query_before total = 313.4M  (<1B OK)`.

- [ ] **Step 6: Commit.**

```bash
git add configs/worldmodel/openvla_oft_input_token_chunk.yaml tests/unit_tests/test_config_validation.py
git commit -m "feat(wm): lean-debottleneck query_before OFT predictor (~313M)"
```

## Task 9: SFT a full-data discrete OpenVLA-OFT on libero_goal

`use_l1_regression=false` selects the discrete merged-LM-head policy (no L1 action
head). `experiment=openvla_oft_hdf5` is the full-data SFT route (the
`*_one_trajectory*` experiments are the cold-start variants).

**Files:**
- Use: `scripts/train_vla.sh`, `configs/experiment/openvla_oft_hdf5.yaml`, `configs/VLA/openvla_oft.yaml`
- Output: `${DVLA_DATA_ROOT}/checkpoints/<full-discrete-oft-goal>/`

- [ ] **Step 1: Launch full-data discrete SFT** (long GPU job; set `out_dir`/`run_tag`):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_vla.sh \
  experiment=openvla_oft_hdf5 task=goal \
  task.openvla_oft.use_l1_regression=false \
  run_tag=full_discrete_goal
```

- [ ] **Step 2: Confirm the route is discrete + full-data before the long run** with `dry_run`:

```bash
bash scripts/train_vla.sh experiment=openvla_oft_hdf5 task=goal \
  task.openvla_oft.use_l1_regression=false print_config=true dry_run=true
```

Expected: resolved config shows `use_l1_regression: false` and a full-data
`sft_dataset_target` (not the one-trajectory dataset).

- [ ] **Step 3: Verify the produced checkpoint is discrete** (no L1 action head component):

```bash
CKPT=$(ls -dt ${DVLA_DATA_ROOT}/outputs/vla/*full_discrete_goal*/ 2>/dev/null | head -1)
ls "${CKPT}" | grep -E "action_head--.*_checkpoint.pt" && echo "L1 (WRONG)" || echo "discrete (OK)"
test -f "${CKPT}/dataset_statistics.json" && echo "stats OK"
```

Expected: `discrete (OK)` and `stats OK`. Record `CKPT` for the next task.

## Task 10: Extract reward HDF5 + both sidecars from existing demos (no rollout)

This consumes the offline LIBERO demos (`processed_data/.../no_noops_t_256`), not
collected rollouts. The reward HDF5 supplies the WM reward-head targets;
`preprocess_oft_action_hidden` runs the trained VLA forward over those demos and
emits the action-hidden (query_after) AND input-token (query_before) sidecars in
one pass.

**Files:**
- Use: `scripts/preprocess/prepare_libero_data.sh`, `dreamervla/preprocess/preprocess_oft_action_hidden.py`

- [ ] **Step 1: Build the reward HDF5 from existing demos** (no rollout):

```bash
bash scripts/preprocess/prepare_libero_data.sh task=goal only=[10_hdf5_reward]
```

Expected: `${DVLA_DATA_ROOT}/processed_data/<artifact>/no_noops_t_256_remaining_reward/*.hdf5`.

- [ ] **Step 2: Extract BOTH sidecars with the full-data discrete VLA**:

```bash
RW="${DVLA_DATA_ROOT}/processed_data/<artifact>/no_noops_t_256_remaining_reward"
AH="${DVLA_DATA_ROOT}/processed_data/<artifact>/no_noops_t_256_full_discrete_action_hidden_h1"
IT="${DVLA_DATA_ROOT}/processed_data/<artifact>/no_noops_t_256_full_discrete_input_token_h1"

CUDA_VISIBLE_DEVICES=0 python -m dreamervla.preprocess.preprocess_oft_action_hidden \
  hdf5_dir="${RW}" \
  out_action_dir="${AH}" \
  out_input_token_dir="${IT}" \
  oft_ckpt="${CKPT}" \
  policy_mode=discrete
```

- [ ] **Step 3: Verify sidecar shapes** (query_after = 56*4096; query_before = 512*4096):

```bash
python - <<'PY'
import h5py, glob, os
for tag, d in [("action_hidden", os.environ["AH"]), ("input_token", os.environ["IT"])]:
    p = sorted(glob.glob(f"{d}/*.hdf5"))[0]
    with h5py.File(p, "r") as h:
        k = list(h["data"])[0]
        ds = h["data"][k]["obs_embedding"]
        print(tag, ds.shape, ds.dtype)
PY
```

Expected: action_hidden `(T, 229376)` (= 56*4096) and input_token `(T, 2097152)`
(= 512*4096), both `float16`.

## Task 11: Train the query_after WM (ceiling, ~610M) on full-data sidecars

`oft_discrete_token_world_model_dinowm_chunk` resolves the action-hidden
(query_after) route on `openvla_oft_action_chunk` (the balanced 610M profile). Point
the dataset at the Task-10 outputs and the Task-9 checkpoint.

**Files:**
- Use: `configs/experiment/oft_discrete_token_world_model_dinowm_chunk.yaml`, `configs/worldmodel/openvla_oft_action_chunk.yaml`

- [ ] **Step 1: Train.**

```bash
CUDA_VISIBLE_DEVICES=0 python -m dreamervla.train \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  logger=tensorboard \
  task.openvla_oft.ckpt_path="${CKPT}" \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${AH}" \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/wm_ceiling/query_after_goal"
```

- [ ] **Step 2: Confirm the WM is the balanced profile and runs without shape errors.**

Expected: TensorBoard `train/` shows `next_latent_mse` / `next_latent_cosine_loss`
decreasing; `resolved_config.yaml` shows `world_model.dim_head=256`,
`mlp_dim=4096`, `num_hist=3`, `latent_stage=query_after`.

## Task 12: Train the query_before WM (ceiling, ~313M) on full-data sidecars

`oft_world_model_dinowm_chunk_input_tokens` resolves the input-token (query_before)
route on `openvla_oft_input_token_chunk` (the 313M profile from Task 8). query_before
latents are identical for discrete/L1 (pre-action-query vision tokens), so the same
extracted input-token sidecar is correct.

**Files:**
- Use: `configs/experiment/oft_world_model_dinowm_chunk_input_tokens.yaml`, `configs/worldmodel/openvla_oft_input_token_chunk.yaml`

- [ ] **Step 1: Train.**

```bash
CUDA_VISIBLE_DEVICES=0 python -m dreamervla.train \
  experiment=oft_world_model_dinowm_chunk_input_tokens \
  task=OpenVLA_Onetraj_LIBERO \
  logger=tensorboard \
  task.openvla_oft.ckpt_path="${CKPT}" \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.input_token_hidden_dir="${IT}" \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/wm_ceiling/query_before_goal"
```

- [ ] **Step 2: Confirm.** Expected: `resolved_config.yaml` shows
`world_model.dim_head=128`, `mlp_dim=2048`, `latent_stage=query_before`,
`token_count=512`; loss curves decreasing.

## Task 13: Measure the ceiling (closed-loop latent diagnostic)

`dreamervla/diagnostics/eval_chunkwm_closeloop.py` rolls the trained WM forward
under the autoregressive recursion (the same `predict_next_chunk` sliding window)
and reports next-latent MSE / cosine vs rollout horizon. Run it on BOTH ceiling
checkpoints; the resulting curve is the upper-bound estimate.

**Files:**
- Use: `dreamervla/diagnostics/eval_chunkwm_closeloop.py`

- [ ] **Step 1: Run the diagnostic on the query_after WM checkpoint** (point it at the
WM checkpoint + the matching sidecar/reward dirs). Record next-latent MSE/cosine at
horizons up to `chunk_rollout_chunks*chunk_size = 32`.

- [ ] **Step 2: Run the diagnostic on the query_before WM checkpoint.**

- [ ] **Step 3: Record the ceiling table** in this document: per scheme, next-latent
MSE/cosine at horizons {1, 8, 16, 32}, model size, and which scheme generalizes
better. This is the headline result.

Expected: a comparison of query_after (610M, dense 168-token latent) vs query_before
(313M, 1536-token latent) showing the WM accuracy ceiling each latent affords.

## Task 14 (optional): Scale to object / spatial / 10

Repeat Tasks 9-13 with `task=object|spatial|10`. Note `time_horizon`/`token_count`
differ by suite (spatial/10 use larger horizons), but `model_dim` and the predictor
profile are unchanged. Log any per-suite throughput knob changes; do not change the
predictor sizing without re-recording the ceiling.

## Pipeline Verification Matrix

- [ ] Task 8 config/test edit: `test_config_validation.py` all pass; query_before instantiates at ~313M (<1B).
- [ ] Task 9 dry-run shows `use_l1_regression=false` + full-data dataset; output checkpoint has no `action_head--*` component.
- [ ] Task 10 sidecar shapes are `(T,229376)` and `(T,2097152)`, float16.
- [ ] Tasks 11-12 `resolved_config.yaml` shows the expected per-scheme profile; loss decreases.
- [ ] Task 13 ceiling table recorded for both schemes.
- [ ] `git diff --check` clean; `py_compile` clean for any touched Python.

## Task 15: Discrete bridge + online wiring for query_before — IMPLEMENTED 2026-06-19

Not needed for the offline ceiling probe (Tasks 9-13 use no actor). This enables
the query_before route in the **online** WMPO actor-critic loop with a discrete
VLA. As of 2026-06-19 it is **implemented in code** (Steps 1-3 below done):
`LatentToOpenVLADiscreteTokenActor` (`dreamervla/models/actor/latent_to_openvla_discrete_token_actor.py`,
extends `OpenVLADiscreteTokenActor`) provides the discrete LM-head bridge;
`obs_to_input_token_embedding` + the `backbone_latent` branch in `OnlineCotrainRunner`
wire the online env (`env.obs_hidden_source=input_token_embedding`); and
`configs/dreamervla/openvla_oft_input_token_wmpo_outcome.yaml` now uses
`head_type: oft_discrete_token`. Remaining: Step 4 (dedicated tests).

**Files:**
- Modify: `dreamervla/models/actor/latent_to_action_hidden_actor.py`
- Modify: the online env (`DreamerVLAOnlineTrainEnv`) + `dreamervla/runners/online_utils.py`
- Modify: `configs/dreamervla/openvla_oft_input_token_wmpo_outcome.yaml`
- Test: `tests/unit_tests/` actor + an online query_before smoke

- [x] **Step 1: Add a discrete LM-head bridge.** Done as a dedicated class
  `LatentToOpenVLADiscreteTokenActor` (extends `OpenVLADiscreteTokenActor`): after
  the Action-Query + TransformerDecoder bridge produces `action_hidden`, the OpenVLA
  LM-head categorical decoder maps it to action tokens (no L1 head). (The older L1
  adapter `LatentToActionHiddenActor` is left intact for L1 checkpoints.)

- [x] **Step 2: Config** `configs/dreamervla/openvla_oft_input_token_wmpo_outcome.yaml`
  now uses `_target_: LatentToOpenVLADiscreteTokenActor`, `head_type: oft_discrete_token`,
  `init_lm_head_ckpt`, `vocab_size`, `action_token_bins`. Composes + `validate_cfg` OK.

- [x] **Step 3: Wire the online input-token obs source.** `obs_to_input_token_embedding`
  (`dreamervla/runners/online_utils.py`) + the `backbone_latent` branch in
  `OnlineCotrainRunner` set `env.obs_hidden_source=input_token_embedding`; the old
  `NotImplementedError` path is gone.

- [x] **Step 4: Tests** — `tests/unit_tests/test_latent_to_openvla_discrete_token_actor.py`
  (4 tests, all pass): the discrete bridge builds and decodes a backbone latent to
  `[B, time_horizon, action_dim]` (flat + tokenized inputs), exposes an OpenVLA LM
  head with no L1 head, and the `openvla_oft_input_token_wmpo_outcome` route wires
  the discrete actor + lean ~313M WM + the online input-token obs source
  (`obs_to_input_token_embedding`) — i.e. no more `NotImplementedError`.

- [ ] **Step 5: Commit** (when the user asks).
