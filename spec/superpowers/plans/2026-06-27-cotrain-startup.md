# Cotrain Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cotrain startup path launch reliably for 0-5 requested GPUs, with a short runnable smoke that follows `spec/99_manual_notes.md` as the source architecture contract.

**Architecture:** Keep the current Hydra launcher and `OnlineCotrainRayRunner` boundaries instead of introducing a new training design. The implementation treats `spec/99_manual_notes.md` as the target naming and topology contract, fixes the startup planner and placement overrides needed to reach that path, and verifies short runs through existing tiny Ray cotrain plus OpenVLA-OFT launcher command generation.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Ray worker groups, pytest, existing DreamerVLA launcher/config/test patterns.

---

## Current Findings

- `spec/99_manual_notes.md` defines the target cotrain roles as `LearnerGroup`, `ActorGroup`, `RolloutGroup`, and `EnvGroup`; current code exposes compatible but older names (`learner`, `infer`/rollout, `envs`, replay-heavy loop).
- `dreamervla.launchers.coldstart_warmup_cotrain` already builds sync and async cotrain commands, but `ngpu=0` is normalized through `max(1, ngpu)` in scaling helpers, so it is not a clear CPU/osmesa path.
- `profile=smoke mode=ray` currently emits profile `env.num_workers=2` and then appends auto-scale `env.num_workers=4`; Hydra's last value wins, so smoke startup is accidentally larger than intended.
- Targeted baseline tests pass with `PYTHONPATH=$PWD`: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`, `test_manual_resource_config_groups.py`, and `test_online_cotrain_ray_runner.py`.
- Local data/checkpoints exist under `data/`, including OpenVLA-OFT traj1 checkpoints, LIBERO datasets, and collected rollout shards.

## Implementation Status

- Implemented launcher GPU-count semantics for `ngpu=0-5`.
- Preserved smoke profile worker counts and explicit override precedence.
- Added 0-GPU Ray/OSMesa CPU overrides for collect, sync warmup, and async online.
- Rejected unsupported `mode=noray ngpu=0` early because the no-Ray OFT collector uses CUDA directly.
- Added Ray collect CPU inference device support with node placement.
- Ran targeted unit tests, Hydra downstream compose/validation, and a short tiny Ray cotrain run.

## Files

- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
  - Owns `ngpu`, render backend, sync/async cotrain command construction, and Ray online placement overrides.
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
  - Add regression coverage for 0-5 GPU startup planning, smoke profile preservation, and async placement command generation.
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py`
  - Add narrow coverage only if runner-side CPU placement behavior needs a direct test.
- Modify only if needed: `spec/99_manual_notes.md`
  - Allowed edits are naming alignment only. Do not change logic, flow, or design intent.

---

### Task 1: Pin Launcher GPU Count Semantics

**Files:**
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`

- [ ] **Step 1: Add failing tests for `ngpu=0` CPU/osmesa planning**

Add tests near the existing launcher scaling tests:

```python
def test_ngpu_zero_does_not_emit_torchrun_or_gpu_ray_placement(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=0,
        launcher_cfg={**_launcher_cfg(), "cotrain_engine": "async", "render_backend": "osmesa"},
    )

    assert "torch.distributed.run" not in plan.cotrain_cmd
    assert "torch.distributed.run" not in plan.cotrain_warmup_cmd
    assert "cluster.component_placement.env=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement.rollout=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement.actor=0" not in plan.cotrain_online_cmd
    assert "cluster.component_placement=null" in plan.cotrain_online_cmd
    assert "inference.placement.strategy=node" in plan.cotrain_online_cmd
    assert "learner.placement.strategy=node" in plan.cotrain_online_cmd
    assert "learner.train_cfg.device=cpu" in plan.cotrain_online_cmd
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ngpu_zero_does_not_emit_torchrun_or_gpu_ray_placement -q
```

Expected: FAIL because the async command does not yet emit the CPU placement overrides.

- [ ] **Step 3: Implement explicit `ngpu` normalization helpers**

In `dreamervla/launchers/coldstart_warmup_cotrain.py`, add small helpers near `_scaled_profile_count`:

```python
def _requested_gpu_count(ngpu: int | None) -> int:
    count = int(ngpu or 0)
    if count < 0:
        raise ValueError(f"ngpu must be >= 0, got {ngpu!r}")
    return count


def _scale_gpu_count(ngpu: int | None) -> int:
    return max(1, _requested_gpu_count(ngpu))
```

Use `_requested_gpu_count` for launch decisions (`torchrun` only when `> 1`) and `_scale_gpu_count` only for workload counts that need at least one worker.

- [ ] **Step 4: Add CPU async online overrides for `ngpu=0`**

Update `_ray_online_scale_overrides` so `ngpu=0` and `render_backend=osmesa` appends:

```python
[
    "cluster.component_placement=null",
    "inference.placement.strategy=node",
    "learner.placement.strategy=node",
    "learner.train_cfg.device=cpu",
    "learner.train_cfg.precision=fp32",
    "env.num_workers=1",
]
```

Also reject `ngpu=0 render_backend=egl` with:

```python
raise ValueError("render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0")
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ngpu_zero_does_not_emit_torchrun_or_gpu_ray_placement -q
```

Expected: PASS.

---

### Task 2: Preserve Smoke Profile Scale

**Files:**
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`

- [ ] **Step 1: Add failing test for Ray smoke collection worker count**

Add:

```python
def test_ray_smoke_profile_num_workers_is_not_overwritten_by_ngpu_autoscale(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="smoke",
        ngpu=1,
    )

    env_worker_overrides = [
        item for item in plan.collect_cmd if item.startswith("env.num_workers=")
    ]
    assert env_worker_overrides == ["env.num_workers=2"]
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ray_smoke_profile_num_workers_is_not_overwritten_by_ngpu_autoscale -q
```

Expected: FAIL because the command currently contains both `env.num_workers=2` and `env.num_workers=4`.

- [ ] **Step 3: Make auto-scale respect profile overrides**

In `build_pipeline_plan`, when deciding `ray_env_scale`, include rendered profile, common, and collect overrides in the explicit override check:

```python
collect_profile_items = _render_overrides(collect_profile_cfg, context)
explicit_collect_overrides = [
    *collect_profile_items,
    *_control_overrides(...),
    *common_overrides,
    *collect_overrides,
]
if selected_mode == "ray" and not _has_override(explicit_collect_overrides, "env.num_workers"):
    ray_env_scale = [f"env.num_workers={_scale_gpu_count(selected_ngpu) * 4}"]
```

Keep ordering so direct `collect.num_workers` and `collect_overrides` still win.

- [ ] **Step 4: Run focused smoke-scale test**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ray_smoke_profile_num_workers_is_not_overwritten_by_ngpu_autoscale -q
```

Expected: PASS.

---

### Task 3: Cover 1-5 GPU Async Placement From Manual Notes

**Files:**
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`

- [ ] **Step 1: Add parameterized tests for async placement**

Add:

```python
@pytest.mark.parametrize("ngpu", [1, 2, 3, 4, 5])
def test_async_egl_placement_scales_with_one_to_five_gpus(tmp_path, ngpu) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = dict(_launcher_cfg())
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "egl"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="multi_gpu",
        ngpu=ngpu,
        launcher_cfg=cfg,
    )

    env_spec = "0" if ngpu == 1 else f"0-{ngpu - 1}"
    assert f"env.num_workers={ngpu}" in plan.cotrain_online_cmd
    assert "env.envs_per_worker=2" in plan.cotrain_online_cmd
    assert f"cluster.component_placement.env={env_spec}" in plan.cotrain_online_cmd
    assert "cluster.component_placement.rollout=0" in plan.cotrain_online_cmd
    assert f"cluster.component_placement.actor={ngpu - 1}" in plan.cotrain_online_cmd
```

- [ ] **Step 2: Run and verify current behavior**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_egl_placement_scales_with_one_to_five_gpus -q
```

Expected: PASS unless Task 1 changes placement helper behavior incorrectly.

- [ ] **Step 3: Add rejection test for `ngpu=0 render_backend=egl`**

Add:

```python
def test_ngpu_zero_rejects_egl_render_backend(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = dict(_launcher_cfg())
    cfg["cotrain_engine"] = "async"
    cfg["render_backend"] = "egl"
    with pytest.raises(ValueError, match="render_backend=egl requires ngpu>=1"):
        build_pipeline_plan(
            mode="ray",
            run_root=tmp_path,
            python="python",
            profile="multi_gpu",
            ngpu=0,
            launcher_cfg=cfg,
        )
```

- [ ] **Step 4: Run focused rejection test**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ngpu_zero_rejects_egl_render_backend -q
```

Expected: PASS after Task 1 implementation.

---

### Task 4: Keep Manual Notes Naming Aligned

**Files:**
- Modify only if needed: `spec/99_manual_notes.md`

- [ ] **Step 1: Scan for inconsistent current names**

Run:

```bash
rg -n "RolloutInferenceWorker|InferenceWorker|LearnerWorker|EnvWorkers\\.EnvWorkers|ActorWorker|RolloutWorker" spec/99_manual_notes.md spec/00_overview.md spec/01_complete_loop.md spec/02_ray.md
```

Expected: report the terms currently present.

- [ ] **Step 2: Decide whether a naming-only edit is needed**

Allowed edits:

```text
InferenceWorker -> RolloutWorker
RolloutInferenceWorker -> RolloutWorker
ActorWorker -> ActorGroup rank / EmbodiedFSDPActor
EnvWorkers.EnvWorkers -> EnvWorker
```

Do not change the described flow, topology, or training logic.

- [ ] **Step 3: If needed, patch only naming text**

Use `apply_patch` for `spec/99_manual_notes.md`. Keep the patch minimal and show only changed names.

- [ ] **Step 4: Verify no broad doc rewrite occurred**

Run:

```bash
git diff -- spec/99_manual_notes.md
```

Expected: either no diff, or a tiny naming-only diff.

---

### Task 5: Verify Short Startup Runs

**Files:**
- No production files expected.

- [ ] **Step 1: Run focused launcher tests**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Ray runner tests**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_manual_resource_config_groups.py tests/unit_tests/test_online_cotrain_ray_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify 0-5 GPU launcher dry-run matrix**

Run:

```bash
for n in 0 1 2 3 4 5; do
  backend=osmesa
  if [ "$n" -gt 0 ]; then backend=egl; fi
  python -m dreamervla.launchers.coldstart_warmup_cotrain \
    mode=ray profile=multi_gpu cotrain_engine=async ngpu="$n" render_backend="$backend" \
    dry_run=true skip_asset_check=true
done
```

Expected: all six commands exit 0. `ngpu=0` uses CPU/osmesa overrides; `ngpu=1..5` emits EGL placement following `GPU0 real env/rollout/learner` and `GPU1..N-1 WMEnv/actor` as far as the current runner supports it.

- [ ] **Step 4: Run no-GPU short cotrain smoke**

Run:

```bash
PYTHONPATH=$PWD python -m dreamervla.train \
  experiment=online_cotrain_ray_world_model_env_tiny \
  training.out_dir=/tmp/dvla-cotrain-ray-wmenv-tiny \
  training.max_steps=1 \
  rollout.steps=2 \
  env.num_workers=1
```

Expected: process starts `OnlineCotrainRayRunner`, performs a short rollout/update path, writes a checkpoint or final metrics, and exits 0.

- [ ] **Step 5: Run OpenVLA-OFT launcher short startup check without long training**

Run with existing collected data and no collection:

```bash
PYTHONPATH=$PWD DVLA_DATA_ROOT=$PWD/data python -m dreamervla.launchers.coldstart_warmup_cotrain \
  mode=ray profile=smoke task=goal ngpu=0 render_backend=osmesa \
  skip_collect=true skip_asset_check=true cotrain_phase=warmup \
  warmup.wm_steps=0 warmup.classifier_steps=0 warmup.replay_epochs=0 warmup.total_env_steps=0 \
  run_root=/tmp/dvla-cotrain-openvla-startup
```

Expected: command reaches the cotrain warmup entry and exits 0, or fails with a concrete environment/dependency issue that is then debugged under `superpowers:systematic-debugging`.

---

### Task 6: Final Review And Reporting

**Files:**
- Inspect all touched files.

- [ ] **Step 1: Review diff**

Run:

```bash
git diff -- dreamervla/launchers/coldstart_warmup_cotrain.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py spec/99_manual_notes.md
```

Expected: changes are scoped to startup planning, tests, and optional naming-only doc alignment.

- [ ] **Step 2: Run final verification commands**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py tests/unit_tests/test_manual_resource_config_groups.py tests/unit_tests/test_online_cotrain_ray_runner.py -q
```

Expected: all pass.

- [ ] **Step 3: Summarize acceptance against user criteria**

Report:

```text
- cotrain startup: command and short smoke evidence
- manual notes compliance: launcher path and naming alignment
- 0-5 GPU support: dry-run matrix evidence
- long training: not run by design
- compatibility: reused current launcher/runner/configs
```

Do not claim completion until fresh verification output has been read.
