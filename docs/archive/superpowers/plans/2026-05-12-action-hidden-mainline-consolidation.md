# Action-Hidden Mainline Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pi0 action-hidden DreamerV3 world-model route the documented current mainline, while marking pooled-hidden, token-WM, LaDiWM, and semantic-bottleneck routes as secondary or legacy.

**Architecture:** Keep code behavior unchanged in this pass. Consolidate README, route docs, config registry, and script registry around `obs -> shared VLA backbone + pi0 action-query block -> action_hidden -> DreamerV3 RSSM`. Preserve old routes as runnable baselines, but remove them from the primary path.

**Tech Stack:** Python, Hydra configs, bash launch wrappers, pytest smoke/unit tests.

---

### Task 1: Document The Current Mainline

**Files:**
- Modify: `README.md`
- Modify: `docs/repository_structure.md`
- Modify: `docs/wm_training_routes.md`

- [ ] **Step 1: Update the project overview**

Replace pooled-hidden/Rynn-pixel language in the overview with the pi0 action-hidden route:

```text
obs + language + state -> Shared VLA Encoder -> pi0 action-query block -> action_hidden -> DreamerV3 RSSM
```

- [ ] **Step 2: Mark the first implementation boundary**

State that the current implemented training path is frozen shared backbone plus precomputed action-hidden sidecar plus WM training. Mark joint finetune and DreamerVLA actor training as follow-up work.

### Task 2: Mark Secondary Routes

**Files:**
- Modify: `configs/README.md`
- Modify: `scripts/README.md`
- Modify: root-level secondary config comments as needed
- Modify: root-level secondary wrapper comments as needed

- [ ] **Step 1: Update registries**

Move these routes out of the current-mainline tables and label them as secondary baselines:

```text
pooled-hidden DreamerV3
DreamerV3 pixel/token baselines
TransDreamer token WM
LaDiWM / Chameleon latent WM
Semantic bottleneck WM
legacy LIBERO-10 archive
```

- [ ] **Step 2: Add lightweight file-level labels**

Add comments to the most visible secondary entry points so shell/config users see the intended status before launching old jobs.

### Task 3: Verify Documentation And Mainline Config

**Files:**
- Test: `configs/rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed.yaml`
- Test: `tests/test_pi0_action_query_head.py`
- Test: `tests/test_preprocess_rynn_pixel_hidden.py`
- Test: `tests/test_pi0_query_pipeline_compat.py`

- [ ] **Step 1: Verify the action-hidden config composes**

Run:

```bash
python -m dreamer_vla.cli.train --config-name rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed --help
```

Expected: exits successfully or prints Hydra/CLI usage without requiring long training.

- [ ] **Step 2: Run action-hidden focused tests**

Run:

```bash
pytest tests/test_pi0_action_query_head.py tests/test_preprocess_rynn_pixel_hidden.py tests/test_pi0_query_pipeline_compat.py -q
```

Expected: all selected tests pass.
