# TransDreamer Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DreamerVLA's TSSM path follow the original TransDreamer temporal alignment and loss targets closely enough to run a fair flat-TSSM comparison.

**Architecture:** Keep the existing DreamerVLA TSSM wrappers, but align the dynamics output to TransDreamer's `post[:, :-1] + action[:, 1:] -> prior` contract. World-model losses consume only target steps `1..T-1`, so image, reward, continue, and hidden reconstruction are computed on the same timesteps as the original implementation. The v4D TSSM configs are adjusted toward TransDreamer defaults without touching non-TSSM variants.

**Tech Stack:** PyTorch, pytest, Hydra YAML configs.

---

### Task 1: Dynamics Temporal Alignment

**Files:**
- Test: `tests/test_tssm_transdreamer_compat.py`
- Modify: `dreamer_vla/models/world_model/tssm_torch.py`

- [ ] **Step 1: Write failing tests**

Add tests that instantiate small `TSSMDynamic` and `TSSMTokenDynamic` modules, call `observe()` with `T=5`, and assert the returned `deter`, `stoch`, `post_logits`, and `prior_logits` length is `T-1`.

- [ ] **Step 2: Verify red**

Run: `/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_tssm_transdreamer_compat.py -q`

Expected: both tests fail because current implementation returns `T` steps.

- [ ] **Step 3: Implement alignment**

Change `TSSMDynamic.observe()` and `TSSMTokenDynamic.observe()` to:
- sample posterior for all `T` observations;
- build the prior from `post_stoch[:, :-1]` and `actions[:, 1:]`;
- return posterior fields trimmed to `[:, 1:]`;
- return prior/deter fields with length `T-1`.

- [ ] **Step 4: Verify green**

Run the same pytest command and confirm it passes.

### Task 2: Loss Target Trimming

**Files:**
- Test: `tests/test_tssm_transdreamer_compat.py`
- Modify: `dreamer_vla/models/world_model/tssm_torch.py`

- [ ] **Step 1: Write failing wrapper tests**

Add small wrapper-model tests asserting `TSSMRynnBackboneWorldModel.loss()` and `TSSMTokenRynnBackboneWorldModel.loss()` can consume a `T=5` batch after sequence outputs become `T-1`.

- [ ] **Step 2: Verify red**

Run: `/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_tssm_transdreamer_compat.py -q`

Expected: wrapper tests fail on target shape mismatch until the loss trims targets.

- [ ] **Step 3: Trim all targets**

In both TSSM wrappers, use `images[:, 1:]`, `obs_embedding[:, 1:]`, `rewards[:, 1:]`, and `dones[:, 1:]` for reconstruction, hidden reconstruction, reward, and continue losses.

- [ ] **Step 4: Verify green**

Run the same pytest command and confirm it passes.

### Task 3: Config Compatibility

**Files:**
- Modify: `configs/dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4d_tssm.yaml`
- Modify: `configs/dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4d_tssm_token.yaml`

- [ ] **Step 1: Adjust TSSM defaults**

Set the TSSM configs closer to original TransDreamer: `act: elu`, `free_nats: 0.0`, `dyn_scale: 0.08`, `rep_scale: 0.02`, `tssm_dropatt: 0.1`. For flat TSSM, use `tssm_layers: 6`, `tssm_d_model: 600`, `tssm_d_inner: 64`, `tssm_d_ff_inner: 1024`, `hidden: 600`.

- [ ] **Step 2: Verify config composition**

Run: `/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_runner_public_api.py::test_all_configs_compose_and_resolve_route_specific_runner_targets -q`

Expected: config composition passes.

### Task 4: Focused Regression

**Files:**
- Existing tests only.

- [ ] **Step 1: Run focused tests**

Run:
- `/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_tssm_transdreamer_compat.py -q`
- `/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_dreamerv3_online_observe.py tests/test_reward_head.py -q`

Expected: all pass.
