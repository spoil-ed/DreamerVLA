# Docker Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `spoil/dreamervla` with pinned DreamerVLA source/environment and two resumable commands for `libero_goal` asset preparation and WM-30/CLS-8/frozen-Dreamer-20000 training.

**Architecture:** A CUDA 12.4/Ubuntu 22.04 image owns immutable source and dependencies under `/opt/dreamervla`; `/data` is the only mutable runtime root. Thin shell scripts select Hydra reproduction configs, while Python validates assets, selects metric checkpoints, persists atomic state, and invokes existing frozen download/preprocess/train entrypoints.

**Tech Stack:** Docker/BuildKit, NVIDIA Container Toolkit, Python 3.11, Hydra/OmegaConf, PyTorch 2.5.1+cu124, Ray 2.55.1, pytest, GitHub Actions, Docker Hub.

## Global Constraints

- Supported profile: Ubuntu 22.04, 8x H100 80 GB, CUDA 12.4.1 userspace, Python 3.11, `libero_goal`.
- OpenVLA source: `Haozhan72/Openvla-oft-SFT-libero-goal-traj1`, revision `d20e1d447dfd87c0daa121b0739e2a379f7fe334`.
- WM runs 30 epochs, CLS runs 8 epochs, frozen-WM/CLS Dreamer runs 20,000 global steps.
- Dreamer consumes the minimum-loss WM checkpoint and maximum-F1 CLS checkpoint.
- Interrupted stages resume from their run roots; validation never deletes user data.
- Do not modify `scripts/install/`, `scripts/download/`, or `scripts/preprocess/`.
- Shell entrypoints contain one Python command; defaults live in Hydra.
- Public image repository: `spoil/dreamervla`.

---

### Task 1: Runtime State, Hash, and Checkpoint Selection

**Files:**
- Create: `tests/unit_tests/test_reproduction_workflow.py`
- Create: `dreamervla/runtime/reproduction.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None`
- Produces: `select_metric_checkpoint(checkpoint_dir: Path, metric_name: str, mode: str) -> SelectedCheckpoint`
- Produces: `decide_stage(state: Mapping[str, Any], stage: str, run_root: Path, budget: int) -> StageDecision`

- [ ] **Step 1: Write failing tests for hashing and minimum/maximum metric selection**

```python
def test_select_metric_checkpoint_uses_minimum_loss(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in ("epoch=0001-loss=0.4.ckpt", "epoch=0002-loss=0.2.ckpt"):
        (root / name).write_bytes(name.encode())
    selected = select_metric_checkpoint(root, metric_name="loss", mode="min")
    assert selected.path.name == "epoch=0002-loss=0.2.ckpt"
```

- [ ] **Step 2: Run focused pytest and observe failure because the module is absent**

Run: `python -m pytest tests/unit_tests/test_reproduction_workflow.py -q`

- [ ] **Step 3: Implement immutable result records, strict flat-name parsing, SHA-256, and atomic JSON writes**

```python
@dataclass(frozen=True)
class SelectedCheckpoint:
    path: Path
    metric_name: str
    value: float
    sha256: str
```

- [ ] **Step 4: Add failing tests for fresh, resume, completed-skip, and mismatch decisions**

- [ ] **Step 5: Implement stage decisions and rerun focused pytest until green**

- [ ] **Step 6: Commit**

```bash
git add dreamervla/runtime/reproduction.py tests/unit_tests/test_reproduction_workflow.py
git commit -s -m "feat: add reproduction workflow state core"
```

### Task 2: Hydra Orchestration and Two Entrypoints

**Files:**
- Modify: `tests/unit_tests/test_reproduction_workflow.py`
- Modify: `tests/unit_tests/test_setup_scripts.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Create: `dreamervla/launchers/reproduce.py`
- Create: `configs/scripts/reproduce/prepare_assets.yaml`
- Create: `configs/scripts/reproduce/train_dreamer.yaml`
- Create: `scripts/reproduce/01_prepare_assets.sh`
- Create: `scripts/reproduce/02_train_dreamer.sh`

**Interfaces:**
- Consumes: Task 1 helpers and existing registered download, preprocess, and train entrypoints.
- Produces: `build_workflow(argv: Sequence[str]) -> ReproductionWorkflow`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Add failing config tests for `libero_goal`, 30, 8, 20000, min loss, max F1, and frozen route assertions**

- [ ] **Step 2: Run focused pytest and verify missing configs/scripts fail**

- [ ] **Step 3: Add static configs and one-command scripts**

```bash
exec python -m dreamervla.launchers.reproduce --config-name reproduce/prepare_assets "$@"
```

- [ ] **Step 4: Add failing tests for command construction, asset reuse, resume, state persistence, and secret redaction**

- [ ] **Step 5: Implement Hydra composition and sequential subprocess orchestration**

The prepare route invokes only `scripts/download_assets.sh` and `scripts/preprocess/prepare_libero_data.sh`. The train route invokes the existing WM, CLS, and cotrain scripts, passes explicit run roots, resumes via `--resume`, validates terminal budgets, and records selected hashes.

- [ ] **Step 6: Run focused tests**

```bash
python -m pytest tests/unit_tests/test_reproduction_workflow.py tests/unit_tests/test_setup_scripts.py tests/unit_tests/test_experiment_stage_scripts.py -q
```

- [ ] **Step 7: Commit**

```bash
git add dreamervla configs/scripts/reproduce scripts/reproduce tests/unit_tests
git commit -s -m "feat: add resumable Dreamer reproduction workflow"
```

### Task 3: Docker Image

**Files:**
- Modify: `tests/unit_tests/test_reproduction_workflow.py`
- Create: `docker/Dockerfile`
- Create: `.dockerignore`
- Create: `tests/e2e_tests/test_docker_reproduction.py`

**Interfaces:**
- Produces: `spoil/dreamervla:cu124-h100-v1` and `/opt/dreamervla/.dreamervla-image.json`.

- [ ] **Step 1: Add a failing Dockerfile contract test for CUDA 12.4.1, Ubuntu 22.04, `/opt/dreamervla`, `/data`, and the existing install workflow**

- [ ] **Step 2: Verify RED because `docker/Dockerfile` is absent**

- [ ] **Step 3: Implement Dockerfile and `.dockerignore`**

Install Miniconda, copy source, invoke `scripts/install_env.sh`, expose the conda env on `PATH`, create image metadata from build args, create `/data`, and verify imports. Exclude `.git`, data, outputs, caches, worktrees, and local `third_party` from build context.

- [ ] **Step 4: Add an opt-in `DVLA_DOCKER_SMOKE=1` e2e test for metadata, imports, and dry-runs**

- [ ] **Step 5: Run focused tests and all shell syntax checks**

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile .dockerignore tests
git commit -s -m "build: add pinned DreamerVLA container image"
```

### Task 4: Public Documentation and Publishing CI

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/README.md`
- Modify: `scripts/README.md`
- Modify: `configs/README.md`
- Create: `docs/docker_reproduction.md`
- Create: `.github/workflows/docker-publish.yml`
- Modify: `tests/unit_tests/test_reproduction_workflow.py`

**Interfaces:**
- Produces exact `docker pull`, prepare, train, resume, and inspection instructions.

- [ ] **Step 1: Add failing tests for documentation links, `spoil/dreamervla`, two scripts, and secret-free CI**

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Write the guide and concise registry links**

- [ ] **Step 4: Add GitHub Actions build/push workflow using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets**

- [ ] **Step 5: Run focused docs/hygiene tests and commit**

```bash
git add .github README.md README.zh-CN.md docs scripts/README.md configs/README.md tests
git commit -s -m "docs: publish Docker reproduction guide"
```

### Task 5: Full Source and Image Verification

**Files:** Modify only when a failing gate exposes a tested defect.

- [ ] **Step 1: Run format, lint, full unit tests, and shell syntax**

```bash
ruff format --check dreamervla tests
ruff check dreamervla tests
python -m pytest tests/unit_tests -q
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

- [ ] **Step 2: Build the release image**

```bash
docker build --progress=plain -f docker/Dockerfile --build-arg DVLA_GIT_COMMIT="$(git rev-parse HEAD)" --build-arg DVLA_IMAGE_VERSION=cu124-h100-v1 -t spoil/dreamervla:cu124-h100-v1 .
```

- [ ] **Step 3: Run install diagnostics and reproduction dry-run inside the image with all GPUs**

- [ ] **Step 4: Inspect image history and metadata for credentials and host paths**

- [ ] **Step 5: For any defect, add a failing regression test, fix it, and rerun every affected gate**

### Task 6: Docker Hub Release

**Files:** No source changes unless verification exposes a tested release defect.

- [ ] **Step 1: Tag the verified image with `v1` and `sha-$(git rev-parse --short=12 HEAD)`**

- [ ] **Step 2: Push `cu124-h100-v1`, `v1`, and the immutable SHA tag**

- [ ] **Step 3: Use `docker buildx imagetools inspect` to prove all tags are public and resolve to one digest**

- [ ] **Step 4: Report source commit, tags, digest, size, verification evidence, and exact public commands**
