# WM Single-Trajectory Overfit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one dry-run-safe experiment command that randomly initializes the configured world model, overfits one LIBERO demo, and proves learning with full-window MSE and cosine-similarity evaluation.

**Architecture:** A focused diagnostic module owns Hydra composition, one-demo HDF5 loading, epoch iteration, convergence tracking, artifacts, and plotting. A thin shell launcher only sets the repository/data environment and invokes the module. Pure helpers and an injected tiny world model keep the training loop testable on CPU without launching a real GPU job.

**Tech Stack:** Python 3.11, PyTorch, Hydra/OmegaConf, h5py, NumPy, matplotlib, pytest, Bash.

---

## File Map

- Create `dreamervla/diagnostics/wm_single_trajectory_overfit.py`: CLI, config composition, HDF5 loading, exact epoch batching, training/evaluation, convergence, checkpoints, summaries, and plot.
- Create `scripts/experiments/wm_single_trajectory_overfit.sh`: thin dry-run-safe launcher.
- Create `tests/unit_tests/test_wm_single_trajectory_overfit.py`: pure helper, tiny-model training, CLI, and launcher tests.
- Modify `scripts/README.md`: register the one-command experiment and its outputs.

### Task 1: Window Coverage, Convergence, And Input Planning

**Files:**
- Create: `tests/unit_tests/test_wm_single_trajectory_overfit.py`
- Create: `dreamervla/diagnostics/wm_single_trajectory_overfit.py`

- [ ] **Step 1: Write failing tests for exact epoch coverage and convergence**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from dreamervla.diagnostics import wm_single_trajectory_overfit as diag


def test_epoch_batches_visit_each_window_once() -> None:
    starts = diag.sliding_window_starts(episode_len=12, sequence_len=5)
    batches = list(
        diag.iter_epoch_batches(
            starts,
            batch_size=3,
            rng=np.random.default_rng(7),
        )
    )

    visited = np.concatenate(batches)
    assert sorted(visited.tolist()) == starts.tolist()
    assert len(set(visited.tolist())) == len(starts)


def test_convergence_requires_consecutive_threshold_passes() -> None:
    tracker = diag.ConvergenceTracker(
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=3,
    )

    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.04, cosine_similarity=0.97) is False
    assert tracker.streak == 0
    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.97) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.98) is True
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py
```

Expected: collection fails because `wm_single_trajectory_overfit` does not exist.

- [ ] **Step 3: Implement the pure helpers and settings types**

Create these public test surfaces in the diagnostic module:

```python
@dataclass
class ConvergenceTracker:
    mse_threshold: float
    cosine_threshold: float
    required_passes: int
    streak: int = 0

    def observe(self, *, mse: float, cosine_similarity: float) -> bool:
        passed = mse <= self.mse_threshold and cosine_similarity >= self.cosine_threshold
        self.streak = self.streak + 1 if passed else 0
        return self.streak >= self.required_passes


def sliding_window_starts(*, episode_len: int, sequence_len: int) -> np.ndarray:
    count = episode_len - sequence_len + 1
    if count <= 0:
        raise ValueError(
            f"episode length {episode_len} is shorter than sequence length {sequence_len}"
        )
    return np.arange(count, dtype=np.int64)


def iter_epoch_batches(
    starts: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    shuffled = rng.permutation(starts)
    for offset in range(0, len(shuffled), batch_size):
        yield shuffled[offset : offset + batch_size]
```

Also add `EpisodeArrays`, `RunSettings`, `parse_args(argv=None)`, and validation for
positive epochs, batch size, evaluation interval, required passes, non-negative MSE,
and cosine thresholds in `[-1, 1]`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit the pure contract**

```bash
git add dreamervla/diagnostics/wm_single_trajectory_overfit.py tests/unit_tests/test_wm_single_trajectory_overfit.py
git commit -s -m "feat: add WM overfit convergence contract"
```

### Task 2: CPU-Tested Overfit Training Loop

**Files:**
- Modify: `tests/unit_tests/test_wm_single_trajectory_overfit.py`
- Modify: `dreamervla/diagnostics/wm_single_trajectory_overfit.py`

- [ ] **Step 1: Add a failing end-to-end CPU test with an injected tiny WM**

```python
import torch


class _TinyWorldModel(torch.nn.Module):
    num_hist = 1
    chunk_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.zeros(2))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = batch["obs_embedding"][:, 1, 0, :]
        prediction = self.prediction[None].expand_as(target)
        mse = torch.nn.functional.mse_loss(prediction, target)
        cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
        return {
            "_loss": mse,
            "hidden_mse": mse.detach(),
            "hidden_cosine_loss": (1.0 - cosine).detach(),
        }


def test_run_overfit_converges_and_writes_checkpoints(tmp_path: Path) -> None:
    episode = diag.EpisodeArrays(
        hidden=np.ones((6, 1, 2), dtype=np.float32),
        lang=np.zeros((4,), dtype=np.float32),
        actions=np.zeros((6, 7), dtype=np.float32),
        rewards=np.zeros((6,), dtype=np.float32),
        proprio=np.zeros((6, 8), dtype=np.float32),
    )
    settings = diag.RunSettings(
        max_epochs=20,
        batch_size=2,
        lr=0.2,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=2,
        seed=3,
    )

    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=episode,
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "converged"
    assert summary["best_hidden_mse"] <= 0.03
    assert summary["best_cosine_similarity"] >= 0.95
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
    assert (tmp_path / "checkpoints" / "final.ckpt").is_file()
```

- [ ] **Step 2: Run only the tiny-model test and verify the missing API failure**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py::test_run_overfit_converges_and_writes_checkpoints
```

Expected: fails because `run_overfit` is not defined.

- [ ] **Step 3: Implement batch construction, weighted evaluation, and epoch training**

The implementation must:

```python
def evaluate_all_windows(
    model: torch.nn.Module,
    episode_tensors: dict[str, torch.Tensor],
    starts: np.ndarray,
    *,
    sequence_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    weighted_loss = weighted_mse = weighted_cosine_loss = 0.0
    sample_count = 0
    with torch.inference_mode(), _autocast(device):
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = make_batch(episode_tensors, batch_starts, sequence_len)
            output = model(batch)
            count = len(batch_starts)
            weighted_loss += float(output["_loss"].float().cpu()) * count
            weighted_mse += float(output["hidden_mse"].float().cpu()) * count
            weighted_cosine_loss += (
                float(output["hidden_cosine_loss"].float().cpu()) * count
            )
            sample_count += count
    return {
        "loss": weighted_loss / sample_count,
        "hidden_mse": weighted_mse / sample_count,
        "cosine_similarity": 1.0 - weighted_cosine_loss / sample_count,
    }
```

`run_overfit` performs an epoch-0 baseline evaluation, then trains every shuffled
window once per epoch with AdamW and gradient clipping. It evaluates every
`eval_every` epochs and at the final epoch, updates `ConvergenceTracker`, writes one
JSONL record per epoch/evaluation, saves `best.ckpt` on lower MSE (cosine breaks ties),
saves `final.ckpt` unconditionally, and returns `status=converged` or
`status=not_converged`. Each epoch prints one concise progress line.

- [ ] **Step 4: Run the unit test and verify real parameter learning**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py::test_run_overfit_converges_and_writes_checkpoints
```

Expected: 1 passed; the test reaches the thresholds using optimizer updates rather
than mocked metrics.

- [ ] **Step 5: Add and pass a maximum-epoch failure test**

```python
def test_run_overfit_reports_not_converged_at_epoch_limit(tmp_path: Path) -> None:
    episode = diag.EpisodeArrays(
        hidden=np.ones((6, 1, 2), dtype=np.float32),
        lang=np.zeros((4,), dtype=np.float32),
        actions=np.zeros((6, 7), dtype=np.float32),
        rewards=np.zeros((6,), dtype=np.float32),
        proprio=np.zeros((6, 8), dtype=np.float32),
    )
    settings = diag.RunSettings(
        max_epochs=1,
        batch_size=2,
        lr=0.0,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.001,
        cosine_threshold=0.99,
        required_passes=2,
        seed=3,
    )
    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=episode,
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert summary["status"] == "not_converged"
    assert summary["epochs_completed"] == 1
```

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the training loop**

```bash
git add dreamervla/diagnostics/wm_single_trajectory_overfit.py tests/unit_tests/test_wm_single_trajectory_overfit.py
git commit -s -m "feat: train WM overfit probe by epoch"
```

### Task 3: Real Hydra/HDF5 CLI, Plot, And One-Command Launcher

**Files:**
- Modify: `tests/unit_tests/test_wm_single_trajectory_overfit.py`
- Modify: `dreamervla/diagnostics/wm_single_trajectory_overfit.py`
- Create: `scripts/experiments/wm_single_trajectory_overfit.sh`
- Modify: `scripts/README.md`

- [ ] **Step 1: Add failing tests for HDF5 loading, dry-run planning, and launcher shape**

Create temporary hidden/raw HDF5 files with `data/demo_0` and assert:

```python
def test_load_episode_reads_matching_hidden_and_raw_demo(tmp_path: Path) -> None:
    hidden_path, raw_path = _write_episode_fixture(tmp_path)
    episode = diag.load_episode(hidden_path, raw_path, "demo_0")
    assert episode.hidden.shape == (6, 1, 2)
    assert episode.actions.shape == (6, 7)
    assert episode.proprio.shape == (6, 8)


def test_experiment_launcher_is_thin_and_dry_run_safe() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "wm_single_trajectory_overfit.sh"
    text = script.read_text(encoding="utf-8")
    assert "dreamervla.diagnostics.wm_single_trajectory_overfit" in text
    assert '"$@"' in text
    assert "--run" not in text.split('"$@"')[0]
```

Also test `build_plan` with explicit fixture paths and verify defaults:
`max_epochs=200`, `eval_every=5`, `MSE=0.03`, `cosine=0.95`, and
`required_passes=3`.

- [ ] **Step 2: Run the new tests and verify expected missing behavior**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py
```

Expected: HDF5/launcher/plan tests fail because those surfaces are incomplete.

- [ ] **Step 3: Implement Hydra composition and real CLI execution**

Compose static config as follows:

```python
register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
    cfg = compose(
        config_name="train",
        overrides=[
            "experiment=openvla_onetraj_libero_cotrain_noray",
            f"task={args.task}",
            "logger=tensorboard",
        ],
    )
OmegaConf.resolve(cfg)
```

Default `task=openvla_onetraj_libero`, HDF5 filename is
`open_the_middle_drawer_of_the_cabinet_demo.hdf5`, and demo key is `demo_0`.
If explicit `--hidden-hdf5`/`--raw-hdf5` are absent, resolve them under
`cfg.task.openvla_oft.input_token_hidden_dir` and `cfg.task.hdf5_reward_dir`.
Instantiate random weights with `hydra.utils.instantiate(cfg.world_model)`; do not
load any checkpoint. `main()` returns 0 for dry-run/converged and 2 for
`not_converged`, while exceptions write `error.txt` and propagate.

- [ ] **Step 4: Implement summaries and non-interactive learning curves**

Write `summary.json` and `summary.md` with baseline, best, final, thresholds, epoch
count, and status. Use matplotlib's `Agg` backend to create three aligned panels for
epoch training loss, evaluation MSE, and evaluation cosine similarity; draw threshold
lines on the latter two and save `overfit_curves.png`.

- [ ] **Step 5: Add the thin launcher and script documentation**

The shell launcher follows existing experiment scripts:

```bash
#!/usr/bin/env bash
# Random-init WM single-trajectory overfit verification. Dry-run unless --run is passed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

"${PYTHON:-python}" -m dreamervla.diagnostics.wm_single_trajectory_overfit "$@"
```

Register the command and artifacts in `scripts/README.md`.

- [ ] **Step 6: Run focused tests, shell syntax, and dry-run**

Run:

```bash
pytest -q tests/unit_tests/test_wm_single_trajectory_overfit.py
bash -n scripts/experiments/wm_single_trajectory_overfit.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/experiments/wm_single_trajectory_overfit.sh \
  --hidden-hdf5 data/datasets/processed_data/libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h1/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
  --raw-hdf5 data/datasets/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward/open_the_middle_drawer_of_the_cabinet_demo.hdf5
```

Expected: tests and syntax check pass; dry-run exits 0, reports
`"initialization": "random"`, and does not create a checkpoint or GPU process.

- [ ] **Step 7: Commit the user-facing experiment**

```bash
git add dreamervla/diagnostics/wm_single_trajectory_overfit.py \
  scripts/experiments/wm_single_trajectory_overfit.sh \
  scripts/README.md \
  tests/unit_tests/test_wm_single_trajectory_overfit.py
git commit -s -m "feat: add one-command WM overfit verification"
```

### Task 4: Final Verification And Push

**Files:**
- Verify all files above; no new production files.

- [ ] **Step 1: Run formatter/linter checks on changed Python files**

```bash
ruff check dreamervla/diagnostics/wm_single_trajectory_overfit.py \
  tests/unit_tests/test_wm_single_trajectory_overfit.py
ruff format --check dreamervla/diagnostics/wm_single_trajectory_overfit.py \
  tests/unit_tests/test_wm_single_trajectory_overfit.py
```

Expected: both commands exit 0.

- [ ] **Step 2: Run focused and adjacent regression tests**

```bash
pytest -q \
  tests/unit_tests/test_wm_single_trajectory_overfit.py \
  tests/unit_tests/test_wm_single_episode_overfit_diagnostic.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Verify no training process was started and inspect the diff**

```bash
ps -eo pid,args | rg 'wm_single_trajectory_overfit.*--run' | rg -v 'rg ' || true
git diff --check
git status --short
git log -5 --oneline --decorate
```

Expected: no matching training process, no whitespace errors, and only intended
commits/files are present.

- [ ] **Step 4: Push the signed commits**

```bash
git push origin main
```

Expected: remote `main` advances to the final local commit. If network access is
unavailable, report the exact push error and leave the verified commits locally.
