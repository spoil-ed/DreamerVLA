# Bilingual Reproduction README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public English and Chinese READMEs with the shortest complete instructions for reproducing DreamerVLA through Docker or a native Conda environment, including asset preparation, staged training, and automatic resume.

**Architecture:** Keep both READMEs as mirrored entry documents with reciprocal language links and identical commands. Reuse the repository-owned install, verification, asset, and training scripts so documentation does not duplicate runtime behavior. Protect the public interface with focused documentation-contract tests, then verify the native environment and dry-run both reproduction stages.

**Tech Stack:** Markdown, Bash, Docker, Conda, pytest

---

## Task 1: Strengthen the public documentation contract

**Files:**
- Modify: `tests/unit_tests/test_reproduction_workflow.py:434`
- Test: `tests/unit_tests/test_reproduction_workflow.py`

- [x] Add assertions that `README.md` and `README.zh-CN.md` link to one another.
- [x] Add assertions that both READMEs document the pinned Docker image, asset script, training script, native installer, native verifier, data root, stage budgets, and resume behavior.
- [x] Run `pytest -q tests/unit_tests/test_reproduction_workflow.py -k public_docs_register` and confirm the old READMEs fail the expanded contract.

## Task 2: Replace the English and Chinese READMEs

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [x] Replace `README.md` with a concise English-first guide containing the Chinese language link, workflow summary, prerequisites, recommended Docker commands, native Conda commands, resume behavior, output locations, and a short explanation of what the image contains.
- [x] Replace `README.zh-CN.md` with the same section order and exact shell commands, translated into concise Chinese and linking back to English.
- [x] Run `pytest -q tests/unit_tests/test_reproduction_workflow.py -k public_docs_register` and confirm the contract passes.

## Task 3: Verify native setup, launch, and resume contracts

**Files:**
- Verify: `scripts/install_env.sh`
- Verify: `scripts/install/60_verify.sh`
- Verify: `scripts/reproduce/01_prepare_assets.sh`
- Verify: `scripts/reproduce/02_train_dreamer.sh`
- Test: `tests/unit_tests/test_reproduction_workflow.py`

- [x] Run `bash scripts/install/60_verify.sh` to verify the existing native `dreamervla` environment and third-party packages.
- [x] Run `conda run -n dreamervla bash scripts/reproduce/01_prepare_assets.sh dry_run=true` and confirm asset download/check commands resolve without mutating production data.
- [x] Run `conda run -n dreamervla bash scripts/reproduce/02_train_dreamer.sh dry_run=true` and confirm WM 30 epochs, classifier 8 epochs, and frozen-WM/CLS Dreamer 20,000-step commands are emitted.
- [x] Run the focused reproduction workflow tests, including the existing checkpoint-resume decision coverage.

## Task 4: Final verification and handoff

**Files:**
- Verify: `README.md`
- Verify: `README.zh-CN.md`
- Verify: `tests/unit_tests/`

- [ ] Search both READMEs for stale placeholders or mismatched commands.
- [ ] Run `pytest -q tests/unit_tests/test_reproduction_workflow.py tests/unit_tests/test_setup_scripts.py`.
- [ ] Run the full unit suite with `pytest -q tests/unit_tests`.
- [ ] Review `git diff --check`, `git diff`, and `git status --short`.
- [ ] Commit the documentation and test changes with a signed-off Conventional Commit.
