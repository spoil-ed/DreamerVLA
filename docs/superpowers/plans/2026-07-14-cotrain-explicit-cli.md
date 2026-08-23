# Cotrain Explicit CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use an execution workflow to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--config openvla_libero --wm_ckpt PATH --cls_ckpt PATH` to the cotrain launcher and make `openvla_libero` the canonical Hydra experiment name.

**Architecture:** Keep `train.sh` as a transparent one-command shell entrypoint. Parse friendly options only in `dreamervla.launchers.cotrain`, translate them to Hydra overrides, then reuse the existing composition, checkpoint-contract inference, environment construction, and subprocess launch paths. Preserve raw Hydra overrides and checkpoint environment variables for compatibility.

**Tech Stack:** Python 3.11, argparse, Hydra/OmegaConf, pytest, Bash.

---

### Task 1: Pin the public CLI contract with failing tests

**Files:**
- Modify: `tests/unit_tests/test_cotrain_launcher.py`

- [ ] **Step 1: Add tests for friendly option translation**

Add tests that create empty WM/CLS checkpoint files and call:

```python
launch = build_launch(
    [
        "--config",
        "openvla_libero",
        "--wm_ckpt",
        str(wm),
        f"--cls_ckpt={classifier}",
        "manual_cotrain.global_steps=3",
    ]
)
assert "experiment=openvla_libero" in launch.command
assert f"init.world_model_state_ckpt={json.dumps(str(wm.resolve()))}" in launch.command
assert f"init.classifier_state_ckpt={json.dumps(str(classifier.resolve()))}" in launch.command
assert launch.cfg.manual_cotrain.global_steps == 3
```

Add focused tests asserting that exactly one friendly checkpoint flag, a nonexistent
path, and a friendly flag duplicated by its Hydra key each raise `ValueError` or
`FileNotFoundError` with the offending option named in the message.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
conda run -n dreamervla pytest -q tests/unit_tests/test_cotrain_launcher.py
```

Expected: the new cases fail because `_overrides` currently rejects the public flags.

### Task 2: Implement the Python option adapter

**Files:**
- Modify: `dreamervla/launchers/cotrain.py`
- Test: `tests/unit_tests/test_cotrain_launcher.py`

- [ ] **Step 1: Parse public options and retain Hydra overrides**

Add an `argparse.ArgumentParser` with optional `--config`, `--wm_ckpt`, and
`--cls_ckpt` arguments. Use `parse_known_args` so remaining values continue through
`_overrides`. Translate them in `build_launch`:

```python
experiment = args.config or DEFAULT_EXPERIMENT
values.insert(0, f"experiment={experiment}")
values.append(f"init.world_model_state_ckpt={_hydra_string(wm_path)}")
values.append(f"init.classifier_state_ckpt={_hydra_string(cls_path)}")
```

Set `DEFAULT_EXPERIMENT = "openvla_libero"`. Resolve paths with
`Path.expanduser().resolve()`, accept files or Hugging Face directories, and reject
duplicate friendly/Hydra inputs. Reuse `_component_overrides` for environment fallback
and atomic pair validation.

- [ ] **Step 2: Run launcher tests**

Run:

```bash
conda run -n dreamervla pytest -q tests/unit_tests/test_cotrain_launcher.py
```

Expected: all launcher tests pass, including legacy environment coverage.

### Task 3: Rename the experiment and update active references

**Files:**
- Create: `configs/experiment/openvla_libero.yaml`
- Delete: `configs/experiment/dreamervla_wmcls_cotrain.yaml`
- Modify: `AGENTS.md`
- Modify: `spec/00_overview.md`
- Modify: `spec/04_complete_loop.md`
- Modify: `spec/06_routes.md`
- Modify: `configs/README.md`
- Modify: `scripts/README.md`
- Modify: `docs/repository_structure.md`
- Modify: `docs/reference/routes.md`
- Modify: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Modify: `tests/unit_tests/test_runner_public_api.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`
- Modify: `tests/unit_tests/test_openvla_traj1_libero_matrix.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`

- [ ] **Step 1: Rename the static Hydra recipe without changing its contents**

Move the exact recipe body to `configs/experiment/openvla_libero.yaml`. Do not add an
alias config: the old public name is removed so the registry has one canonical cotrain
experiment.

- [ ] **Step 2: Replace active references and document the command**

Replace active uses of `dreamervla_wmcls_cotrain` with `openvla_libero`. Update the
scripts registry example to:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

Retain historical references under `docs/superpowers/specs/` and
`docs/superpowers/plans/` because they document earlier states.

- [ ] **Step 3: Run composition and repository contract tests**

Run:

```bash
conda run -n dreamervla pytest -q \
  tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py \
  tests/unit_tests/test_runner_public_api.py \
  tests/unit_tests/test_repository_hygiene.py
```

Expected: all selected tests pass and Hydra composes `experiment=openvla_libero` into
`dreamervla.runners.CotrainRunner`.

### Task 4: Verify the shell surface and commit

**Files:**
- Modify only files listed in Tasks 1-3.

- [ ] **Step 1: Dry-run the exact user command**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 COTRAIN_DRY_RUN=1 \
conda run --no-capture-output -n dreamervla \
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt data/outputs/wm_precision_40step_20260714/ckpt/wm_warmup.ckpt \
  --cls_ckpt data/outputs/pre_mainline/classifier/20260713_cls_official/checkpoints/best_window_f11.0000_th0.25.ckpt
```

Expected: exit code 0 and the printed downstream command contains the new experiment
and both resolved `init.*_state_ckpt` overrides.

- [ ] **Step 2: Run formatting and diff checks**

Run:

```bash
conda run -n dreamervla ruff check dreamervla/launchers/cotrain.py tests/unit_tests/test_cotrain_launcher.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Commit the implementation**

```bash
git add AGENTS.md configs dreamervla/launchers/cotrain.py scripts/README.md spec docs \
  tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py \
  tests/unit_tests/test_repository_hygiene.py \
  tests/unit_tests/test_runner_public_api.py
git commit -s -m "feat: add explicit cotrain checkpoint CLI"
```
