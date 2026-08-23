# Docker Layer Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make documentation and ordinary source edits rebuild only DreamerVLA's small final image layer while preserving the complete pinned runtime and public release tags.

**Architecture:** Bootstrap the install workflow from a deliberately small set of dependency inputs, run the expensive environment installation before the full repository copy, then copy all source and perform final verification. Configure GitHub Actions BuildKit caching so the same dependency layer can be restored on clean runners.

**Tech Stack:** Docker BuildKit, Docker Hub, GitHub Actions, pytest, Bash

---

### Task 1: Add failing Docker layer-boundary contracts

**Files:**
- Modify: `tests/unit_tests/test_reproduction_workflow.py:385`
- Test: `tests/unit_tests/test_reproduction_workflow.py`

- [x] **Step 1: Add the dependency/source boundary test**

Add a test that reads `docker/Dockerfile`, locates `bash scripts/install_env.sh` and
`COPY . /opt/dreamervla`, and asserts that the full source copy occurs later. Assert
that `pyproject.toml`, `requirements.txt`, the install scripts/config, and minimal
workflow modules are copied before installation, while `README.md` is not copied
before installation.

```python
def test_dockerfile_caches_dependencies_before_copying_full_source() -> None:
    text = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    install_index = text.index("bash scripts/install_env.sh")
    full_copy_index = text.index("COPY . /opt/dreamervla")
    dependency_prefix = text[:install_index]

    for required in (
        "COPY pyproject.toml requirements.txt",
        "COPY scripts/install_env.sh",
        "COPY scripts/install/",
        "COPY configs/scripts/install/",
        "COPY dreamervla/__init__.py dreamervla/config_resolvers.py",
        "COPY dreamervla/launchers/__init__.py dreamervla/launchers/workflow.py",
    ):
        assert required in dependency_prefix
    assert "COPY README.md" not in dependency_prefix
    assert full_copy_index > install_index
    assert text.count("COPY . /opt/dreamervla") == 1
    assert text.rindex("python -m dreamervla.diagnostics.verify_install") > full_copy_index
```

- [x] **Step 2: Extend the publish workflow contract**

Add assertions to `test_docker_publish_workflow_uses_secrets_and_release_tags` for:

```python
assert "cache-from: type=gha" in text
assert "cache-to: type=gha,mode=max" in text
```

- [x] **Step 3: Run the focused test and verify RED**

Run:

```bash
conda run -n dreamervla pytest -q \
  tests/unit_tests/test_reproduction_workflow.py \
  -k 'dockerfile or docker_publish_workflow'
```

Expected: the new ordering test fails because the current Dockerfile has
`COPY . /opt/dreamervla` before `scripts/install_env.sh`, and the workflow cache
assertions fail because no GHA cache is configured.

### Task 2: Reorder the Dockerfile around stable dependency inputs

**Files:**
- Modify: `docker/Dockerfile:44-75`
- Test: `tests/unit_tests/test_reproduction_workflow.py`

- [x] **Step 1: Copy only install inputs before dependency installation**

Replace the pre-install full source copy with explicit copies for:

```dockerfile
COPY pyproject.toml requirements.txt /opt/dreamervla/
COPY scripts/install_env.sh /opt/dreamervla/scripts/install_env.sh
COPY scripts/install/ /opt/dreamervla/scripts/install/
COPY configs/scripts/install/ /opt/dreamervla/configs/scripts/install/
COPY dreamervla/__init__.py dreamervla/config_resolvers.py /opt/dreamervla/dreamervla/
COPY dreamervla/launchers/__init__.py dreamervla/launchers/workflow.py /opt/dreamervla/dreamervla/launchers/
RUN touch README.md
```

These files are the complete input surface needed by `scripts/install_env.sh` and
the editable package metadata during installation.

- [x] **Step 2: Install dependencies without importing the not-yet-copied application**

Keep the existing cache mounts and install command, but remove the early
`python -m dreamervla.diagnostics.verify_install` call. Retain Conda cleanup and
removal of `/opt/dreamervla-build-data` in the dependency layer.

- [x] **Step 3: Copy full source and perform final verification**

After the dependency `RUN`, add:

```dockerfile
COPY . /opt/dreamervla
```

Keep `.dreamervla-image.json`, `/data`, and
`python -m dreamervla.diagnostics.verify_install` in the following final `RUN` so
OCI metadata still corresponds to the complete source image.

- [x] **Step 4: Run the focused Dockerfile tests and verify GREEN**

Run the focused command from Task 1. Expected: Dockerfile tests pass; workflow cache
assertions remain failing until Task 3.

### Task 3: Persist BuildKit cache in GitHub Actions

**Files:**
- Modify: `.github/workflows/docker-publish.yml:41-57`
- Test: `tests/unit_tests/test_reproduction_workflow.py`

- [x] **Step 1: Add cache import and export to the image action**

Add these keys under `docker/build-push-action@v6`:

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

- [x] **Step 2: Run the focused contracts**

Run the Task 1 pytest command. Expected: all selected tests pass.

- [x] **Step 3: Run static review**

Run:

```bash
git diff --check
git diff -- docker/Dockerfile .github/workflows/docker-publish.yml \
  tests/unit_tests/test_reproduction_workflow.py
```

Expected: no whitespace errors; the full source copy appears exactly once and after
the install `RUN`.

### Task 4: Prove documentation-only cache reuse with real builds

**Files:**
- Temporarily modify and restore: `README.md`
- Verify: `docker/Dockerfile`

- [x] **Step 1: Build the optimized image once**

Run a local BuildKit build tagged `dreamervla:layer-cache-test` with the current
commit, image version `layer-cache-test`, and an explicit UTC build time. Expected:
the dependency layer completes and the final install diagnostic passes.

- [x] **Step 2: Add a temporary Markdown cache probe with `apply_patch`**

Append this comment to `README.md`:

```markdown
<!-- docker-source-layer-cache-probe -->
```

- [x] **Step 3: Rebuild and inspect BuildKit output**

Repeat the build with tag `dreamervla:layer-cache-probe`. Expected: the
`scripts/install_env.sh` dependency-install step is `CACHED`; only
`COPY . /opt/dreamervla` and the final metadata/verification step rerun.

- [x] **Step 4: Restore README and verify a clean diff**

Remove the cache-probe comment with `apply_patch`, then run `git diff --check` and
confirm README contains no probe marker.

### Task 5: Verify, commit, and publish

**Files:**
- Verify: `docker/Dockerfile`
- Verify: `.github/workflows/docker-publish.yml`
- Verify: `tests/unit_tests/`

- [x] **Step 1: Run focused and full tests**

Run:

```bash
conda run -n dreamervla pytest -q \
  tests/unit_tests/test_reproduction_workflow.py \
  tests/unit_tests/test_setup_scripts.py
conda run -n dreamervla pytest -q tests/unit_tests
```

Expected: focused suites and full unit suite pass, with only declared skips.

- [ ] **Step 2: Commit and push source**

Create a signed-off Conventional Commit:

```bash
git add docker/Dockerfile .github/workflows/docker-publish.yml \
  tests/unit_tests/test_reproduction_workflow.py \
  docs/superpowers/plans/2026-07-17-docker-layer-caching.md
git commit -s -m "build: cache docker dependency layers"
git push origin main
```

- [ ] **Step 3: Build and push release tags**

Build with the new commit labels and push:

```text
spoil/dreamervla:cu124-h100-v1
spoil/dreamervla:v1
spoil/dreamervla:sha- followed by the new commit's first 12 characters
```

- [ ] **Step 4: Verify remote publication**

Inspect all three Docker Hub manifests and confirm they share one digest. Pull or
inspect the immutable tag and confirm OCI revision equals the GitHub `origin/main`
commit. Confirm `git status --short` is empty.
