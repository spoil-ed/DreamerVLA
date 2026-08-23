# Unified Train Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dedicated cotrain Python launcher with one generic training launcher extended by a Hydra-selected cotrain contract.

**Architecture:** `dreamervla.launchers.train` owns experiment discovery, generic parsing, resume mapping, Hydra composition, environment construction, command creation, and execution. A `LaunchContract` protocol supplies opt-in CLI normalization, derived overrides, validation, environment checks, and summary lines; `configs/experiment/openvla_libero.yaml` selects `CotrainLaunchContract`.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch checkpoint inspection, pytest, Bash.

---

### Task 1: Specify the Unified Public Entry Point

**Files:**
- Modify: `tests/unit_tests/test_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Test: `tests/unit_tests/test_cotrain_launcher.py`

- [ ] **Step 1: Migrate cotrain tests to the generic builder**

Replace the import with:

```python
from dreamervla.launchers.train import build_launch
```

Add a helper that selects the cotrain experiment:

```python
def _build_cotrain_launch(argv: list[str]) -> ExperimentLaunch:
    return build_launch(["--config", "openvla_libero", *argv])
```

Use `_build_cotrain_launch` in every cotrain test while preserving all existing
checkpoint, resume, classifier-head, environment, and GPU assertions.

- [ ] **Step 2: Add structural single-entrypoint assertions**

```python
def test_cotrain_train_script_uses_unified_train_launcher() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/experiments/cotrain/train.sh").read_text(encoding="utf-8")
    assert "dreamervla.launchers.train" in text
    assert "dreamervla.launchers.cotrain" not in text
    assert not (root / "dreamervla/launchers/cotrain.py").exists()
```

- [ ] **Step 3: Run the migrated tests and verify RED**

```bash
pytest -q tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  -k 'cotrain_launcher or cotrain_train_script_uses_unified'
```

Expected: collection or assertion failure because `train.build_launch` and the
unified script route do not exist yet.

### Task 2: Introduce the Launch Contract Boundary

**Files:**
- Create: `dreamervla/launchers/contracts.py`
- Test: `tests/unit_tests/test_cotrain_launcher.py`

- [ ] **Step 1: Define the contract protocol and no-op implementation**

```python
class LaunchContract(Protocol):
    def normalize_argv(self, argv: Sequence[str]) -> list[str]: ...
    def derive_overrides(self, cfg: DictConfig, overrides: Sequence[str]) -> list[str]: ...
    def validate(self, cfg: DictConfig) -> None: ...
    def update_env(self, cfg: DictConfig, env: dict[str, str]) -> None: ...
    def summary_lines(self, cfg: DictConfig, env: Mapping[str, str]) -> list[str]: ...


class DefaultLaunchContract:
    def normalize_argv(self, argv: Sequence[str]) -> list[str]:
        return list(argv)

    def derive_overrides(self, cfg: DictConfig, overrides: Sequence[str]) -> list[str]:
        return []

    def validate(self, cfg: DictConfig) -> None:
        return None

    def update_env(self, cfg: DictConfig, env: dict[str, str]) -> None:
        return None

    def summary_lines(self, cfg: DictConfig, env: Mapping[str, str]) -> list[str]:
        return []
```

- [ ] **Step 2: Move cotrain-specific rules behind `CotrainLaunchContract`**

Move the public `--wm_ckpt`/`--cls_ckpt` parser, environment compatibility,
checkpoint output-dimension inspection, BCE/CE derivation, frozen-component
validation, GPU topology validation, and summary construction from
`launchers/cotrain.py`. The contract must not compose Hydra or spawn subprocesses.

- [ ] **Step 3: Run contract-focused tests**

```bash
pytest -q tests/unit_tests/test_cotrain_launcher.py
```

Expected at this intermediate step: tests may still fail because the generic
launcher does not instantiate the new contract; the contract module imports and
unit helpers must collect successfully.

### Task 3: Refactor the Generic Launcher into a Reusable Builder

**Files:**
- Modify: `dreamervla/launchers/train.py`
- Test: `tests/unit_tests/test_cotrain_launcher.py`
- Test: `tests/unit_tests/test_experiment_stage_scripts.py`

- [ ] **Step 1: Add `ExperimentLaunch` and experiment discovery**

```python
@dataclass(frozen=True)
class ExperimentLaunch:
    experiment: str
    command: tuple[str, ...]
    env: dict[str, str]
    cfg: DictConfig
    dry_run: bool
    summary_lines: tuple[str, ...] = ()
```

Add `_experiment_from_argv()` that reads `--config`, `--config-name`, or an
`experiment=<name>` override without rejecting contract-specific options.

- [ ] **Step 2: Instantiate the Hydra-selected contract**

Compose the selected experiment once, read `launch.contract`, and instantiate its
`_target_` with `hydra.utils.instantiate`. When absent, construct
`DefaultLaunchContract`. Do not branch on experiment names or runner targets.

- [ ] **Step 3: Add `build_launch(argv)`**

Implement the phase ordering from the design:

```python
experiment = _experiment_from_argv(argv)
discovery_cfg = _compose(experiment, [])
contract = _build_contract(discovery_cfg)
normalized = contract.normalize_argv(argv)
experiment, launcher, overrides = _parse_args(normalized)
cfg = _compose(experiment, overrides)
# apply generic aliases and contract-derived overrides, recomposing as needed
contract.validate(cfg)
env = _build_env(cfg, launcher, data_root=data_root)
contract.update_env(cfg, env)
command = _command(cfg, launcher, experiment, overrides)
return ExperimentLaunch(...)
```

Keep generic resume logic in `_target_overrides` and expose
`__all__ = ["ExperimentLaunch", "build_launch", "main"]`.

- [ ] **Step 4: Make `main()` print and execute the resolved launch**

`main()` calls only `build_launch`, prints generic and contract summary lines,
returns zero for `dry_run`, and otherwise executes the command with the resolved
environment.

- [ ] **Step 5: Run launcher tests and verify GREEN**

```bash
pytest -q tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  -k 'launcher or cotrain_eval_protocol or cotrain_train_script'
```

Expected: all selected tests pass.

### Task 4: Wire Hydra and Remove the Dedicated Route

**Files:**
- Modify: `configs/experiment/openvla_libero.yaml`
- Modify: `scripts/experiments/cotrain/train.sh`
- Delete: `dreamervla/launchers/cotrain.py`
- Delete: `scripts/experiments/dreamer/train.sh`
- Delete: `tests/unit_tests/test_dreamer_train_script.py`
- Modify: `scripts/README.md`
- Modify: `tests/unit_tests/test_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Modify: `tests/unit_tests/test_setup_scripts.py`

- [ ] **Step 1: Select the cotrain contract from Hydra**

Add this launch section to `openvla_libero.yaml`:

```yaml
launch:
  distributed: false
  ngpu: ${manual_cotrain.ngpu}
  gpus: null
  data_root: ${oc.env:DVLA_DATA_ROOT,${dvla.data_root}}
  write_libero_config: true
  required_target_values: []
  contract:
    _target_: dreamervla.launchers.contracts.CotrainLaunchContract
  env:
    PYTHONUNBUFFERED: 1
    HYDRA_FULL_ERROR: 1
    NCCL_NVLS_ENABLE: 0
    RAY_DEDUP_LOGS: 0
```

- [ ] **Step 2: Switch the shell route and delete `cotrain.py`**

The shell command becomes:

```bash
exec python -m dreamervla.launchers.train "$@"
```

Delete the old module after its specialized behavior is covered by contract tests.
Also delete the obsolete Dreamer shell alias and its dedicated test, then remove
that alias from the script registry. Keeping it would leave a tracked shell script
that imports the deleted Python module.

- [ ] **Step 3: Run focused config and script suites**

```bash
pytest -q tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_setup_scripts.py \
  -k 'launcher or release_shell_entrypoints or cotrain'
```

Expected: all selected tests pass. Tests requiring unavailable optional model
dependencies remain deselected.

### Task 5: Verify, Commit, Integrate, and Push

**Files:**
- Verify all files changed in Tasks 1-4
- Create: `docs/superpowers/plans/2026-07-15-unified-train-launcher.md`

- [ ] **Step 1: Run formatting and static checks**

```bash
ruff check dreamervla/launchers/train.py dreamervla/launchers/contracts.py \
  tests/unit_tests/test_cotrain_launcher.py tests/unit_tests/test_experiment_stage_scripts.py
ruff format --check dreamervla/launchers/train.py dreamervla/launchers/contracts.py \
  tests/unit_tests/test_cotrain_launcher.py tests/unit_tests/test_experiment_stage_scripts.py
bash -n scripts/experiments/cotrain/train.sh
git diff --check
```

- [ ] **Step 2: Run the complete relevant verification set**

```bash
pytest -q tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  tests/unit_tests/test_setup_scripts.py \
  -k 'launcher or release_shell_entrypoints or cotrain'
```

- [ ] **Step 3: Commit on the isolated branch**

```bash
git add dreamervla/launchers/train.py dreamervla/launchers/contracts.py \
  dreamervla/launchers/cotrain.py configs/experiment/openvla_libero.yaml \
  scripts/experiments/cotrain/train.sh tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_experiment_stage_scripts.py \
  docs/superpowers/plans/2026-07-15-unified-train-launcher.md
git commit -s -m "refactor: unify training launch contracts"
```

- [ ] **Step 4: Merge into `main`, re-run verification, and push**

Merge the isolated branch without discarding the main worktree's unrelated
changes. Re-run the focused verification from the main workspace, then push
`main` to `origin`.
