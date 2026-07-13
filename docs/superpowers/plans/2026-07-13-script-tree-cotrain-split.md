# Script Tree Simplification and Cotrain Split Implementation Plan

> Superseded scope note: retain the classifier-training,
> single-trajectory-overfit, and world-model-training experiment scripts. Shell
> defaults move to Hydra configs, and cotrain eval uses the required Hydra
> override `eval.ckpt_path=<checkpoint>`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `scripts/` to protected install/download/preprocess entrypoints plus independent cotrain `train.sh` and `eval.sh` commands.

**Architecture:** Replace the trainable WM/CLS periodic-eval recipe with a train-only Hydra recipe. Reuse the existing Ray cotrain launcher for training and the existing LIBERO evaluation launcher for explicit checkpoint evaluation; shell files only establish environment defaults and forward Hydra overrides.

**Tech Stack:** Bash, Python 3.11, Hydra/OmegaConf, Ray, PyTorch, pytest.

## Global Constraints

- Preserve every file under `scripts/download/`, `scripts/install/`, and `scripts/preprocess/` byte-for-byte.
- Preserve `scripts/download_assets.sh`, `scripts/install_env.sh`, and `scripts/preprocess_libero.sh` byte-for-byte.
- Keep only `scripts/experiments/cotrain/train.sh` and `scripts/experiments/cotrain/eval.sh` as experiment shell entrypoints.
- Random initialization applies only to the world model and classifier; the VLA still loads the task-selected OpenVLA-OFT checkpoint.
- Require both warm-start component paths or neither; reject a partial pair.
- Require an explicit `COTRAIN_CKPT` for evaluation and never auto-select a checkpoint.
- Do not modify `spec/99_manual_notes.md`.
- Use `apply_patch` for repository file edits and preserve unrelated user changes.

---

## File Map

**Create**

- `configs/dreamervla/wmcls_cotrain_ray.yaml`: trainable staged cotrain with periodic evaluation disabled.
- `configs/experiment/dreamervla_wmcls_cotrain_ray.yaml`: public train-only experiment composition.
- `scripts/experiments/cotrain/train.sh`: eight-GPU cotrain launcher.
- `scripts/experiments/cotrain/eval.sh`: explicit-checkpoint 100-episode LIBERO evaluator.

**Modify**

- `dreamervla/launchers/frozen_model_cotrain_ray.py`: recognize the train-only recipe and make WM/CLS paths optional only for that recipe.
- `tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`: cover random, paired, and partial initialization.
- `tests/unit_tests/test_experiment_stage_scripts.py`: specify the two-script cotrain experiment surface.
- `tests/unit_tests/test_setup_scripts.py`: specify the reduced release shell tree.
- `tests/unit_tests/test_repository_hygiene.py`: point active documentation assertions at the new entrypoints.
- `tests/unit_tests/test_manual_resource_config_groups.py`, `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`, `tests/unit_tests/test_runner_public_api.py`: remove assertions whose only subject is a deleted shell wrapper.
- Active repository guidance and command references, including `AGENTS.md`, `CLAUDE.md`, `README.md`, `README.zh-CN.md`, `SETUP.md`, `scripts/README.md`, `configs/README.md`, `docs/README.md`, `docs/install.md`, `docs/data_layout.md`, `docs/repository_structure.md`, `docs/reference/routes.md`, `docs/reference/model_datasets/openvla_oft_libero_goal.md`, `docs/tutorials/experiments/README.md`, `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`, `docs/tutorials/experiments/EXPLAINED.md`, and `spec/00_overview.md`, `spec/01_goal.md`, `spec/05_ray_runtime.md`, `spec/06_routes.md`, `spec/09_frozen_model_pre_mainline.md`.
- Python messages/docstrings that name deleted scripts: `dreamervla/runners/dreamervla_runner.py`, `dreamervla/runners/embodied_eval_runner.py`, and `dreamervla/runners/pretokenize_vla_runner.py`.

**Delete**

- `configs/dreamervla/wmcls_cotrain_ray_eval.yaml`.
- `configs/experiment/dreamervla_wmcls_cotrain_ray_eval.yaml`.
- Every shell file outside the protected set and the two new cotrain files.

---

### Task 1: Train-Only Recipe and Optional WM/CLS Warm Start

**Files:**
- Create: `configs/dreamervla/wmcls_cotrain_ray.yaml`
- Create: `configs/experiment/dreamervla_wmcls_cotrain_ray.yaml`
- Modify: `dreamervla/launchers/frozen_model_cotrain_ray.py`
- Modify: `tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`
- Delete: `configs/dreamervla/wmcls_cotrain_ray_eval.yaml`
- Delete: `configs/experiment/dreamervla_wmcls_cotrain_ray_eval.yaml`

**Interfaces:**
- Consumes: `build_launch(argv: list[str]) -> FrozenRayLaunch` and existing `ManualCotrainRayRunner` behavior where an empty learner init mapping leaves WM/CLS randomly initialized.
- Produces: public experiment `dreamervla_wmcls_cotrain_ray`; `build_launch` accepts no WM/CLS environment paths for this experiment, accepts both, and rejects exactly one.

- [ ] **Step 1: Add failing launcher tests for the three initialization states**

Add tests equivalent to:

```python
def test_wmcls_train_launcher_uses_random_components_when_pair_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run"))

    launch = build_launch(["experiment=dreamervla_wmcls_cotrain_ray"])
    cfg = launcher._compose_training_config(launch.command)

    assert not any(item.startswith("init.world_model_state_ckpt=") for item in launch.command)
    assert not any(item.startswith("init.classifier_state_ckpt=") for item in launch.command)
    assert launch.periodic_eval.enabled is False
    assert cfg.manual_cotrain.learner_updates_enabled is True


def test_wmcls_train_launcher_rejects_partial_component_pair(
    tmp_path: Path, monkeypatch
) -> None:
    wm = tmp_path / "wm.ckpt"
    wm.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(ValueError, match="both WORLD_MODEL_CKPT and CLASSIFIER_CKPT"):
        build_launch(["experiment=dreamervla_wmcls_cotrain_ray"])
```

Retarget every trainable-WM/CLS occurrence in this test module from
`dreamervla_wmcls_cotrain_ray_eval` to `dreamervla_wmcls_cotrain_ray`, including
resume, debug-schedule, and config-composition parametrizations. Keep both
frozen experiment names unchanged. In the paired test, assert both quoted init
overrides remain present.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q \
  tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py::test_wmcls_train_launcher_uses_random_components_when_pair_is_absent \
  tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py::test_wmcls_train_launcher_rejects_partial_component_pair
```

Expected: failures because `dreamervla_wmcls_cotrain_ray` is unsupported and missing paths are still mandatory.

- [ ] **Step 3: Add the train-only Hydra recipe**

Create `configs/experiment/dreamervla_wmcls_cotrain_ray.yaml` with:

```yaml
# @package _global_
defaults:
  - /task: openvla_onetraj_libero
  - override /dreamervla: wmcls_cotrain_ray
  - override /classifier: openvla_oft_spatial
  - _self_

runner:
  logger:
    logger_backends: [tensorboard, wandb]
    wandb_mode: offline
    wandb_proxy: null
```

Create `configs/dreamervla/wmcls_cotrain_ray.yaml` from the current staged
trainable recipe, preserving all model, replay, learner, placement, and staged
policy settings while changing only:

```yaml
training:
  out_dir: ${oc.env:RUN_ROOT,${oc.env:DVLA_DATA_ROOT,${oc.env:DVLA_ROOT,.}/data}/outputs/pre_mainline/wmcls_cotrain_ray/${now:%Y%m%d_%H%M%S}}

manual_cotrain:
  eval_interval_global_steps: 0
  eval_initial_global_step: false
```

Remove the two `_eval` recipe files after the replacement contains every other
key from the old files.

- [ ] **Step 4: Implement optional atomic warm-start resolution**

In `dreamervla/launchers/frozen_model_cotrain_ray.py`, introduce:

```python
TRAINABLE_WMCLS_EXPERIMENT = "dreamervla_wmcls_cotrain_ray"


def _component_assignment_pair(experiment: str) -> tuple[str | None, str | None]:
    wm_value = os.environ.get("WORLD_MODEL_CKPT", "").strip()
    classifier_value = os.environ.get("CLASSIFIER_CKPT", "").strip()
    if experiment == TRAINABLE_WMCLS_EXPERIMENT:
        if bool(wm_value) != bool(classifier_value):
            raise ValueError(
                "set both WORLD_MODEL_CKPT and CLASSIFIER_CKPT for a warm start, "
                "or leave both unset for random initialization"
            )
        return (wm_value or None, classifier_value or None)
    return (
        _required_environment_path("WORLD_MODEL_CKPT"),
        _required_environment_path("CLASSIFIER_CKPT"),
    )
```

Add the constant to the supported set. Resolve checkpoint files only when
returned values are not `None`, and append the two `init.*_state_ckpt` overrides
only when both resolved paths exist. Update `_default_out_dir` and log prefixes
to use `wmcls_cotrain_ray`. Do not weaken frozen-model requirements.

- [ ] **Step 5: Run focused launcher/config tests and verify GREEN**

Run:

```bash
pytest -q tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py \
  tests/unit_tests/test_frozen_model_pre_mainline_config.py
```

Expected: all tests pass with the train-only experiment and frozen routes intact.

- [ ] **Step 6: Commit Task 1**

```bash
git add configs/dreamervla configs/experiment \
  dreamervla/launchers/frozen_model_cotrain_ray.py \
  tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py
git commit -s -m "refactor: separate train-only wmcls cotrain recipe"
```

---

### Task 2: Independent Cotrain Train and Eval Scripts

**Files:**
- Create: `scripts/experiments/cotrain/train.sh`
- Create: `scripts/experiments/cotrain/eval.sh`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`

**Interfaces:**
- Consumes: `dreamervla.launchers.frozen_model_cotrain_ray` with experiment `dreamervla_wmcls_cotrain_ray`; `dreamervla.launchers.train --config-name eval_libero_vla`.
- Produces: `train.sh` accepting optional paired `WORLD_MODEL_CKPT`/`CLASSIFIER_CKPT`; `eval.sh` requiring `COTRAIN_CKPT`.

- [ ] **Step 1: Replace experiment-script tests with failing cotrain contract tests**

Replace shell-wrapper tests at the start of
`tests/unit_tests/test_experiment_stage_scripts.py` with:

```python
def test_cotrain_experiment_directory_contains_train_and_eval() -> None:
    root = Path(__file__).resolve().parents[2]
    cotrain = root / "scripts" / "experiments" / "cotrain"
    assert sorted(path.name for path in cotrain.iterdir()) == [
        "eval.sh",
        "train.sh",
    ]


def test_cotrain_train_script_uses_train_only_recipe_without_pinned_warm_states() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/experiments/cotrain/train.sh").read_text()
    assert "dreamervla.launchers.frozen_model_cotrain_ray" in text
    assert "experiment=dreamervla_wmcls_cotrain_ray" in text
    assert "manual_cotrain.global_steps" in text
    assert "/inspire/" not in text
    assert "20260712" not in text


def test_cotrain_eval_script_requires_explicit_policy_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/experiments/cotrain/eval.sh").read_text()
    assert "COTRAIN_CKPT" in text
    assert "eval.ckpt_kind=vla_policy" in text
    assert "eval.num_episodes_per_task=10" in text
    assert "eval.num_envs=25" in text
    assert "eval.require_strict_component_load=true" in text
```

Add a subprocess test that runs `eval.sh` with `COTRAIN_CKPT` absent and asserts
exit code 2 plus `COTRAIN_CKPT=/path/to/manual_cotrain.ckpt is required` on
stderr.

- [ ] **Step 2: Run the new script tests and verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_experiment_stage_scripts.py -k 'cotrain or experiment_directory'
```

Expected: failures because `scripts/experiments/cotrain/` does not exist.

- [ ] **Step 3: Create the thin training script**

Create executable `scripts/experiments/cotrain/train.sh` with repository/data
root setup, optional conda activation, eight-GPU and NCCL defaults, and:

```bash
WMCLS_COTRAIN_GLOBAL_STEPS="${WMCLS_COTRAIN_GLOBAL_STEPS:-20000}"

exec "${PYTHON_EXECUTABLE}" -m dreamervla.launchers.frozen_model_cotrain_ray \
  experiment=dreamervla_wmcls_cotrain_ray \
  manual_cotrain.global_steps="${WMCLS_COTRAIN_GLOBAL_STEPS}" \
  "$@"
```

Do not assign WM/CLS checkpoint variables in the script.

- [ ] **Step 4: Create the explicit-checkpoint evaluation script**

Create executable `scripts/experiments/cotrain/eval.sh`. After normal root,
Python, conda, OSMesa, and GPU setup, enforce:

```bash
if [[ -z "${COTRAIN_CKPT:-}" ]]; then
  echo "COTRAIN_CKPT=/path/to/manual_cotrain.ckpt is required" >&2
  exit 2
fi
if [[ ! -f "${COTRAIN_CKPT}" ]]; then
  echo "COTRAIN_CKPT is not a file: ${COTRAIN_CKPT}" >&2
  exit 2
fi
```

Then execute `dreamervla.launchers.train --config-name eval_libero_vla` with
these defaults before `"$@"`:

```bash
experiment=eval_libero_vla
eval.ckpt_path="${COTRAIN_CKPT}"
eval.ckpt_kind=vla_policy
init.vla_ckpt_path="${BASE_VLA_CKPT}"
eval.task_suite_name=libero_goal
eval.task_ids='[0,1,2,3,4,5,6,7,8,9]'
eval.num_episodes_per_task=10
eval.num_envs=25
eval.action_steps=8
eval.history_length=1
eval.seed=7
eval.scheme=rlinf_chunk
eval.enumerate_all_init_states=false
eval.reconfigure_per_episode=true
eval.action_postprocess=openvla_oft
eval.require_strict_component_load=true
eval.render_backend=osmesa
eval.cotrain_diagnostics=true
eval.cotrain_expected_trajectories=100
eval.cotrain_encode_batch_size=4
```

Default `BASE_VLA_CKPT` to the canonical goal one-trajectory checkpoint under
`${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/`. Default the output below
`${DVLA_DATA_ROOT}/outputs/eval/cotrain/` with a timestamp.

Set both new files executable:

```bash
chmod +x scripts/experiments/cotrain/train.sh scripts/experiments/cotrain/eval.sh
```

- [ ] **Step 5: Run script tests and shell syntax checks**

Run:

```bash
pytest -q tests/unit_tests/test_experiment_stage_scripts.py -k 'cotrain or experiment_directory'
bash -n scripts/experiments/cotrain/train.sh scripts/experiments/cotrain/eval.sh
```

Expected: tests pass and `bash -n` exits 0.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/experiments/cotrain tests/unit_tests/test_experiment_stage_scripts.py
git commit -s -m "feat: split cotrain train and eval entrypoints"
```

---

### Task 3: Remove Obsolete Shell Entrypoints and Update Hygiene Tests

**Files:**
- Delete: every non-protected shell entrypoint outside `scripts/experiments/cotrain/`
- Modify: `tests/unit_tests/test_setup_scripts.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`
- Modify: `tests/unit_tests/test_manual_resource_config_groups.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_runner_public_api.py`
- Modify or delete shell-only cases in other tests returned by the reference scan

**Interfaces:**
- Consumes: exact protected and cotrain script set from Tasks 1-2.
- Produces: 22 shell files: 4 download, 7 install, 6 preprocess, 3 protected top-level wrappers, and 2 cotrain scripts.

- [ ] **Step 1: Change the curated-tree test first**

Update `test_release_scripts_tree_is_curated` to expect top-level files:

```python
{"README.md", "download_assets.sh", "install_env.sh", "preprocess_libero.sh"}
```

and directories:

```python
{"download", "experiments", "install", "preprocess"}
```

Also assert recursively that the exact shell set equals the protected scripts
and two cotrain files. Remove test cases whose only subject is a deleted shell
wrapper, while keeping underlying Python launcher, workflow, runner, config, and
diagnostic tests.

Add the exclusive experiment-directory assertion at this stage:

```python
experiments = root / "scripts" / "experiments"
assert sorted(path.name for path in experiments.iterdir()) == ["cotrain"]
```

- [ ] **Step 2: Run hygiene tests and verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_setup_scripts.py::test_release_scripts_tree_is_curated \
  tests/unit_tests/test_experiment_stage_scripts.py::test_cotrain_experiment_directory_contains_train_and_eval
```

Expected: failures listing old shell files and experiment directories.

- [ ] **Step 3: Delete obsolete shell files with `apply_patch`**

Delete all old top-level shell files except the three protected wrappers; delete
`scripts/eval/`; delete the three old experiment directories. Do not touch any
file under a protected step directory.

- [ ] **Step 4: Remove stale shell-only assertions**

Run this reference scan:

```bash
rg -n 'check_ray\.sh|collect_parallel\.sh|e2e_[A-Za-z0-9_]+\.sh|eval_libero_vla\.sh|run_wandb_relay_sync\.sh|start_ray\.sh|train_dreamervla\.sh|experiments/(classifier_training|single_trajectory_overfit|world_model_training)' tests
```

Remove only assertions about deleted shell surfaces. Retarget documentation
assertions to the new train/eval paths. Keep tests for underlying code.

- [ ] **Step 5: Verify tree tests and protected-file invariants**

Run:

```bash
pytest -q tests/unit_tests/test_setup_scripts.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_repository_hygiene.py \
  tests/unit_tests/test_manual_resource_config_groups.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py \
  tests/unit_tests/test_runner_public_api.py
git diff --exit-code c2780cb -- scripts/download scripts/install scripts/preprocess \
  scripts/download_assets.sh scripts/install_env.sh scripts/preprocess_libero.sh
```

Expected: tests pass and the protected-file diff emits no output.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts tests/unit_tests
git commit -s -m "refactor: remove obsolete shell entrypoints"
```

---

### Task 4: Rewrite Active Entrypoint Documentation

**Files:**
- Modify: active files listed in the File Map
- Do not modify: `spec/99_manual_notes.md`

**Interfaces:**
- Consumes: final shell paths from Tasks 2-3.
- Produces: active guidance with no command pointing to a deleted shell file.

- [ ] **Step 1: Add or update failing documentation assertions**

Require active registries to contain:

```text
scripts/experiments/cotrain/train.sh
scripts/experiments/cotrain/eval.sh
```

Reject deleted release entrypoint names. Exclude historical
`docs/superpowers/specs/` and `docs/superpowers/plans/` from this scan.

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_repository_hygiene.py -k 'document or entrypoint or script'
```

Expected: failures showing old paths in active docs.

- [ ] **Step 3: Rewrite `scripts/README.md`**

Register only the protected workflow wrappers and numbered steps plus the two
cotrain scripts. Show paired warm-start and random-init training commands, and
the required explicit-checkpoint eval command. Keep the data-layout link.

- [ ] **Step 4: Update active docs and source messages**

Use these canonical commands:

```bash
bash scripts/experiments/cotrain/train.sh
```

```bash
COTRAIN_CKPT=/path/to/manual_cotrain.ckpt \
  bash scripts/experiments/cotrain/eval.sh
```

Where a deleted diagnostic wrapper has no replacement, document its existing
`python -m dreamervla...` entry only when still supported. Preserve valid
architecture text. Replace Python source messages that name deleted wrappers.

- [ ] **Step 5: Scan and test active references**

Run:

```bash
rg -n 'scripts/(check_ray|collect_parallel|e2e_[A-Za-z0-9_]+|eval_libero_vla|run_wandb_relay_sync|start_ray|train_dreamervla)\.sh|scripts/experiments/(classifier_training|single_trajectory_overfit|world_model_training)' \
  AGENTS.md CLAUDE.md README.md README.zh-CN.md SETUP.md configs/README.md scripts/README.md docs spec \
  --glob '!docs/superpowers/specs/**' --glob '!docs/superpowers/plans/**' --glob '!spec/99_manual_notes.md'
pytest -q tests/unit_tests/test_repository_hygiene.py
```

Expected: no stale active paths and all hygiene tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add AGENTS.md CLAUDE.md README.md README.zh-CN.md SETUP.md configs/README.md \
  scripts/README.md docs spec dreamervla/runners tests/unit_tests/test_repository_hygiene.py
git commit -s -m "docs: document reduced cotrain script workflow"
```

---

### Task 5: Full Verification

**Files:**
- Verify only; if a real regression appears, modify its smallest owning file and rerun the focused test first.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: fresh evidence for shell inventory, protected files, config composition, launcher behavior, and unit tests.

- [ ] **Step 1: Check shell syntax and executable bits**

Run:

```bash
bash -n $(rg --files scripts -g '*.sh' | sort)
test -x scripts/experiments/cotrain/train.sh
test -x scripts/experiments/cotrain/eval.sh
```

Expected: exit 0.

- [ ] **Step 2: Verify protected files against the approved-design commit**

Run:

```bash
git diff --exit-code c2780cb -- scripts/download scripts/install scripts/preprocess \
  scripts/download_assets.sh scripts/install_env.sh scripts/preprocess_libero.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest -q tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_setup_scripts.py \
  tests/unit_tests/test_repository_hygiene.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the complete unit suite**

Run:

```bash
pytest -q tests/unit_tests
```

Expected: zero failures.

- [ ] **Step 5: Check formatting, whitespace, inventory, and status**

Run:

```bash
ruff check dreamervla tests
ruff format --check dreamervla tests
git diff --check
rg --files scripts -g '*.sh' | sort
git status --short
```

Expected: lint/format checks exit 0; inventory contains 22 approved shell files;
status contains no unrelated artifacts.

- [ ] **Step 6: Commit verification-only fixes if needed**

If verification required a correction:

```bash
git commit -s -m "fix: complete cotrain script split verification"
```

If no correction was needed, do not create an empty commit.
