# Manual Cotrain Mainline Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the current manual Ray cotrain route satisfy `spec/99_manual_notes.md` for the real OpenVLA/LIBERO mainline, not only CPU stubs.

**Architecture:** Keep the four target groups already introduced: LearnerGroup updates WM/classifier, ActorGroup trains the VLA actor with FSDP, RolloutGroup owns no-grad inference, and EnvGroup owns real/WM stepping. Close the missing real-env path by letting RolloutGroup encode image observations into `obs_embedding` before sampling actions, and by letting EnvGroup attach rollout sidecars to replay transitions while keeping policy actions separate from env postprocessed actions. Make ActorGroup FSDP export enter state-dict collectives on all actor ranks while only rank 0 publishes patches.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Ray WorkerGroup/Channel, PyTorch/FSDP, OpenVLA-OFT extractor interfaces, pytest.

---

### Task 1: Real Env Observation Encoding In RolloutGroup

**Files:**
- Modify: `tests/unit_tests/test_multistep_rollout_worker.py`
- Modify: `dreamervla/workers/rollout/multistep_rollout_worker.py`
- Modify: `dreamervla/workers/inference/_test_rollout_stub.py`

- [x] **Step 1: Write failing test for image observation encoding**

Add a test that constructs `MultiStepRolloutWorker` with an encoder bundle config, passes an `ObservationMsg` whose `obs` has no `obs_embedding` or `latent`, and asserts `generate_once()` emits:
- action chunk from the local policy,
- `forward_inputs["hidden"]` from the encoder-generated hidden,
- optional `lang_emb`,
- per-slot extractor reset on `is_first=True`.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_multistep_rollout_worker.py::test_generate_once_encodes_real_env_observation_without_obs_embedding -q
```
Expected before implementation: FAIL because `ObservationMsg.obs must include obs_embedding or latent`.

- [x] **Step 2: Implement encoder-backed hidden resolution**

In `MultiStepRolloutWorker`, keep the current fast path for `obs_embedding`/`latent`. When missing, require `encoder_cfg`, build it from Hydra-style config, keep one extractor per `ObservationMsg.key`, reset that extractor on `obs["is_first"]`, then support these encoder contracts in order:
- bundle with `make_extractor()` plus `predict_batch(preps)`,
- extractor with `step(obs, task_description)`,
- encoder with `encode_observation(obs, task_description)`,
- callable encoder returning a hidden tensor or mapping.

Return a structured pair `(hidden, extras)` so `lang_emb` can be forwarded without mutating env obs.

- [x] **Step 3: Run rollout worker tests**

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_multistep_rollout_worker.py -q
```
Expected after implementation: PASS.

### Task 2: Replay Sidecars And Real Env Action Postprocess

**Files:**
- Modify: `tests/unit_tests/test_trajectory_env_worker.py`
- Modify: `dreamervla/workers/env/_test_envs.py`
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`

- [x] **Step 1: Write failing replay-sidecar test**

Add a no-embedding real-env test double. Its observations mimic `DreamerVLAOnlineTrainEnv`: image/state/task metadata but no `obs_embedding`. Feed a `RolloutResultMsg` whose `forward_inputs` include `hidden` and `lang_emb`. Assert the completed replay episode contains `obs_embedding`, `lang_emb`, and `policy_version` metadata.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_trajectory_env_worker.py::test_real_env_worker_attaches_rollout_sidecars_to_no_embedding_env_records -q
```
Expected before implementation: FAIL because replay transition lacks sidecars or the env test double raises `KeyError`.

- [x] **Step 2: Write failing action-postprocess test**

Add a test with `env_cfg.action_postprocess=openvla_oft` and an action whose gripper value requires OpenVLA-OFT postprocess. Assert `env.step()` receives the postprocessed env action while replay stores the original policy action separately from the env action.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_trajectory_env_worker.py::test_real_env_worker_postprocesses_openvla_oft_env_action_without_overwriting_policy_action -q
```
Expected before implementation: FAIL because the raw policy action is passed directly to env step.

- [x] **Step 3: Implement EnvWorker sidecar/action handling**

In `BaseTrajectoryEnvWorker.apply_rollout_result()`, pass rollout sidecars into `_step_slot()`. In `_step_slot()`, attach `obs_embedding`, `lang_emb`, `policy_version`, and model versions to the observation copy used for replay transition construction. Add `_env_action_from_policy_action()` so only the environment step sees OpenVLA-OFT gripper postprocessing when `env_cfg.action_postprocess` is `openvla_oft`; `_make_transition()` still receives the original policy action.

- [x] **Step 4: Configure real manual OFT env postprocess**

Set `env.real.cfg.action_postprocess: openvla_oft` in `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`. Do not apply this to WMEnv.

- [x] **Step 5: Run EnvWorker tests**

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_trajectory_env_worker.py -q
```
Expected after implementation: PASS.

### Task 3: FSDP-Safe Actor To Rollout Sync

**Files:**
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `tests/unit_tests/test_embodied_fsdp_actor.py`
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`

- [x] **Step 1: Write failing runner sync test**

Update/add a test proving `ManualCotrainRayRunner._run_global_step()` broadcasts `sync_model_to_rollout()` to the whole ActorGroup, not `execute_on(0)`, so every FSDP rank can enter state-dict collectives. Assert only rollout pull remains a group broadcast.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_run_global_step_syncs_all_actor_ranks_for_fsdp_collective_export -q
```
Expected before implementation: FAIL because runner restricts sync to actor rank 0.

- [x] **Step 2: Write failing actor rank gating test**

Add a test that sets `EmbodiedFSDPActor.rank = 1`, replaces `state_dict()` and `_syncer()` with recording fakes, calls `sync_model_to_rollout()`, and asserts `state_dict()` was called but no patch was pushed. This proves nonzero ranks participate but do not publish.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_embodied_fsdp_actor.py::test_sync_model_to_rollout_nonzero_rank_participates_without_pushing_patch -q
```
Expected before implementation: FAIL because nonzero rank pushes too.

- [x] **Step 3: Implement FSDP state export and rank-0 push**

In `EmbodiedFSDPActor.state_dict()`, detect `FullyShardedDataParallel` and use `FSDP.state_dict_type(... FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))`. In `sync_model_to_rollout()`, always call `state_dict()` but only push when `self.rank == 0` and the state dict is non-empty.

- [x] **Step 4: Update runner checkpoint export**

When saving manual checkpoints, call `ActorGroup.state_dict()` across all actor ranks and choose the first non-empty returned state. This avoids single-rank FSDP collective deadlocks.

- [x] **Step 5: Run actor/runner tests**

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_embodied_fsdp_actor.py tests/unit_tests/test_manual_cotrain_ray_runner.py -q
```
Expected after implementation: PASS.

### Task 4: Config Wiring For Real Manual OFT Rollout Encoding

**Files:**
- Modify: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`

- [x] **Step 1: Write failing config composition test**

Assert the manual OFT config sets `rollout.encoder_cfg` to an OpenVLA-OFT rollout bundle config and carries the task-derived image/history/hidden-source fields.

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_cotrain_oft_rollout_encoder_cfg_uses_oft_bundle -q
```
Expected before implementation: FAIL because `rollout.encoder_cfg` is null.

- [x] **Step 2: Configure encoder bundle**

Add `rollout.encoder_cfg` using `dreamervla.workers.inference.oft_rollout:OFTRolloutBundle`, with `policy_cfg` derived from the existing collect/OFT policy settings, `unnorm_key`, selected image keys, history, rotation, hidden source, action head expectation, proprio expectation, and `device: auto`.

- [x] **Step 3: Run config tests**

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_cotrain_oft_backbone_experiment_composes tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_cotrain_oft_rollout_encoder_cfg_uses_oft_bundle -q
```
Expected after implementation: PASS.

### Task 5: Documentation Delta And Verification

**Files:**
- Modify: `spec/99_manual_notes.md`
- Run tests only for affected modules.

- [x] **Step 1: Add Current vs Target note**

Append under `## Current vs Target` without moving or rewriting user-authored sections. Record that the manual route now has the target code path, and list remaining non-code validation requirements such as full multi-GPU OpenVLA/LIBERO smoke if unavailable in this session.

- [x] **Step 2: Run targeted regression suite**

Run:
```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_multistep_rollout_worker.py tests/unit_tests/test_trajectory_env_worker.py tests/unit_tests/test_embodied_fsdp_actor.py tests/unit_tests/test_manual_cotrain_ray_runner.py tests/unit_tests/test_manual_cotrain_config_validation.py tests/unit_tests/test_manual_cotrain_placement.py tests/unit_tests/test_cotrain_messages.py tests/unit_tests/test_wm_env_bootstrap.py -q
```
Expected: PASS.

- [x] **Step 3: Run tiny manual Ray smoke**

Run:
```bash
WANDB_MODE=offline /home/user01/miniconda3/envs/dreamervla/bin/python -m dreamervla.train experiment=manual_cotrain_ray_tiny training.out_dir=/tmp/dvla_manual_cotrain_tiny_mainline_completion
```
Expected: exits 0 and logs `manual-cotrain` metrics. If Ray local startup fails for external environment reasons, capture the exact error and do not claim this smoke passed.

- [x] **Step 4: Inspect diff**

Run:
```bash
git diff -- dreamervla/workers/rollout/multistep_rollout_worker.py dreamervla/workers/env/trajectory_env_worker.py dreamervla/workers/actor/embodied_fsdp_actor.py dreamervla/runners/manual_cotrain_ray_runner.py configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml tests/unit_tests/test_multistep_rollout_worker.py tests/unit_tests/test_trajectory_env_worker.py tests/unit_tests/test_embodied_fsdp_actor.py tests/unit_tests/test_manual_cotrain_ray_runner.py spec/99_manual_notes.md
```
Expected: changes are scoped to manual-cotrain completion and do not revert unrelated worktree changes.
