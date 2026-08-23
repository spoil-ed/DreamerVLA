# OpenVLA Staged Cotrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the original OpenVLA-OFT encoder and action decoder through a staged real-SFT / WM+CLS / imagined-PPO global step, with step-local data, aligned world-model history, complete checkpointing, and causal evaluation.

**Architecture:** Introduce a full OpenVLA policy adapter with raw and projected-token forward paths, make real completed trajectories a first-class step-local batch, execute explicit phase barriers in `ManualCotrainRayRunner`, and carry WM histories across chunk inference. Keep the legacy random latent bridge limited to frozen feasibility recipes.

**Tech Stack:** Python 3.11, PyTorch/FSDP, Ray, Hydra/OmegaConf, OpenVLA-OFT, LIBERO, pytest.

---

### Task 1: Preserve the periodic-resume repair

**Files:**
- Modify: `dreamervla/launchers/manual_cotrain_vla_eval.py`
- Modify: `dreamervla/launchers/frozen_model_cotrain_ray.py`
- Modify: `tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`

- [x] **Step 1: Add composition regressions for optional resume keys**
- [x] **Step 2: Verify the tests fail with a struct-key composition error**
- [x] **Step 3: Normalize Hydra override prefixes and emit `++manual_cotrain.resume_ckpt`**
- [x] **Step 4: Run the launcher/config regression set**

### Task 2: Add the full OpenVLA policy contract

**Files:**
- Create: `dreamervla/models/embodiment/openvla_oft_policy.py`
- Modify: `dreamervla/models/embodiment/__init__.py`
- Modify: `dreamervla/runners/rollout_hidden_extractor.py`
- Modify: `dreamervla/workers/inference/oft_rollout.py`
- Create: `tests/unit_tests/test_openvla_oft_policy.py`

- [x] **Step 1: Write fake-upstream tests for raw and projected-token parity**

  Assert that raw forward computes projected vision tokens and that latent forward
  accepts those same tokens, native prompt tensors, optional proprio, and exact action
  token IDs. Assert both paths produce identical action logits when given identical
  projected tokens.

- [x] **Step 2: Run the focused tests and verify RED**

  ```bash
  PYTHONPATH=. pytest tests/unit_tests/test_openvla_oft_policy.py -q
  ```

- [x] **Step 3: Implement the common native OpenVLA multimodal/action path**

  Factor the differentiable projected-token-to-action logic currently embedded in
  `OFTBatchedDecoder` into an `nn.Module`. Expose `encode_raw`, `sample_raw`,
  `sample_latent`, and `evaluate_action_tokens`. Do not add a Transformer bridge or
  learned action queries.

- [x] **Step 4: Make the rollout extractor delegate to the shared module**

  Preserve existing collector behavior and checkpoint metadata while allowing the
  trainable module to return prompt tensors and sampled token IDs.

- [x] **Step 5: Run policy, extractor, and checkpoint-loading tests GREEN**

  ```bash
  PYTHONPATH=. pytest tests/unit_tests/test_openvla_oft_policy.py tests/unit_tests/test_rollout_hidden_extractor.py tests/unit_tests/test_openvla_oft_base_eval_runner.py -q
  ```

### Task 3: Route the mainline to the full policy

**Files:**
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`
- Modify: `dreamervla/workers/rollout/multistep_rollout_worker.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Modify: `dreamervla/runners/embodied_eval_runner.py`
- Modify: `tests/unit_tests/test_openvla_traj1_libero_matrix.py`
- Modify: `tests/unit_tests/test_multistep_rollout_worker.py`
- Modify: `tests/unit_tests/test_embodied_fsdp_actor.py`

- [x] **Step 1: Write config and worker tests requiring one full VLA policy**

  Require the mainline policy target to be the new module, forbid a separate encoder
  plus random policy in rollout workers, and require raw/latent mode dispatch.

- [x] **Step 2: Verify the new assertions fail**
- [x] **Step 3: Update Hydra construction and rollout/actor dispatch**

  Real observations use `sample_raw`; WM observations use `sample_latent`. Both return
  exact action-token IDs and prompt conditioning in `forward_inputs`. Actor evaluation
  calls the same native latent path.

- [x] **Step 4: Save and load the complete VLA state**

  Change `vla_policy` evaluation so it restores the full VLA module. Keep legacy
  checkpoint dispatch explicit for old frozen feasibility artifacts.

- [x] **Step 5: Run worker/config/eval tests GREEN**

### Task 4: Introduce a step-local real-trajectory batch

**Files:**
- Modify: `dreamervla/workers/cotrain/messages.py`
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `dreamervla/scheduler/worker_group.py`
- Create: `tests/unit_tests/test_step_local_real_batch.py`

- [x] **Step 1: Write tests for exactly-once completed-episode drain**

  The batch must preserve raw images, task description, proprio, continuous actions,
  action-token IDs, rewards, terminal flags, and success. Draining a global step twice
  returns no old episode.

- [x] **Step 2: Verify RED**
- [x] **Step 3: Add `RealTrajectoryBatch` and worker/group drain APIs**
- [x] **Step 4: Verify serialization and no cross-step retention GREEN**

### Task 5: Add successful-real encoder SFT and re-encoding

**Files:**
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Modify: `dreamervla/scheduler/actor_group.py`
- Create: `tests/unit_tests/test_encoder_sft_phase.py`

- [x] **Step 1: Write tests for parameter masks and no-success skip**

  Require only vision backbone/projector parameters to change in SFT, actor parameters
  to remain unchanged, successful episodes to supply exact action-token labels, and a
  zero-success batch to emit a skip metric.

- [x] **Step 2: Verify RED**
- [x] **Step 3: Implement low-LR multi-epoch encoder SFT**
- [x] **Step 4: Re-encode every real episode after SFT and return CPU sidecars**
- [x] **Step 5: Verify parameter hashes and latent-version metadata GREEN**

### Task 6: Make WM/CLS updates current-step, multi-epoch, and calibrated

**Files:**
- Modify: `dreamervla/workers/actor/learner_worker.py`
- Modify: `dreamervla/workers/replay/replay_worker.py`
- Modify: `dreamervla/algorithms/critic/classifier.py`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`
- Create: `tests/unit_tests/test_step_local_wm_classifier_update.py`

- [x] **Step 1: Write tests that reject stale encoder versions and stale replay**
- [x] **Step 2: Write threshold calibration tests, including single-class fallback**
- [x] **Step 3: Verify RED**
- [x] **Step 4: Add replace-not-append learner data loading**
- [x] **Step 5: Add configured epochs/steps, validation early stopping, and threshold persistence**
- [x] **Step 6: Verify WM/CLS update and calibration metrics GREEN**

### Task 7: Align chunked WM inference history

**Files:**
- Modify: `dreamervla/envs/world_model/latent_world_model_env.py`
- Modify: `tests/unit_tests/test_latent_world_model_env.py`
- Modify: `tests/unit_tests/test_chunk_world_model_closed_loop.py`

- [x] **Step 1: Write a two-chunk stateful model test**

  Make the fake WM output depend on every history slot and prior action. The second
  chunk must receive the exact returned `history` and `actions` from the first chunk.

- [x] **Step 2: Run the focused test and verify RED**
- [x] **Step 3: Store/reset per-slot WM hidden and action histories**
- [x] **Step 4: Commit returned histories after every chunk and preserve async slot order**
- [x] **Step 5: Run all latent-WM environment tests GREEN**

### Task 8: Reorder the manual cotrain global step and enforce one KL transaction

**Files:**
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`
- Create: `tests/unit_tests/test_manual_cotrain_stage_order.py`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`

- [x] **Step 1: Write a call-order test for all stage barriers**

  Require `real collect -> drain -> SFT -> re-encode -> learner load/update -> WM/CLS
  sync -> imagined collect -> actor PPO -> eval/checkpoint` and forbid real shards in
  PPO.

- [x] **Step 2: Write shared cumulative-KL accept/rollback tests**
- [x] **Step 3: Verify RED**
- [x] **Step 4: Split real and imagined interaction phases in the runner**
- [x] **Step 5: Snapshot each causal policy phase, accumulate raw/imagined KL, and roll back the violating phase**
- [x] **Step 6: Verify stage-order, version, and checkpoint tests GREEN**

### Task 9: Add causal trajectory-level evaluation

**Files:**
- Create: `dreamervla/diagnostics/eval_cotrain_transaction.py`
- Modify: `dreamervla/launchers/manual_cotrain_vla_eval.py`
- Modify: `dreamervla/runners/embodied_eval_runner.py`
- Create: `tests/unit_tests/test_cotrain_transaction_eval.py`

- [x] **Step 1: Write deterministic tests for full-trajectory WM recursion and equal trajectory weighting**
- [x] **Step 2: Write classifier metric/undefined-AUC/threshold provenance tests**
- [x] **Step 3: Verify RED**
- [x] **Step 4: Implement fixed 100-trajectory read-only evaluation**
- [x] **Step 5: Persist overall, per-task, horizon, real-CLS, and WM+CLS metrics**
- [x] **Step 6: Verify eval data never enters learner or threshold calibration GREEN**

### Task 10: Full verification and documentation alignment

**Files:**
- Modify: `spec/04_complete_loop.md`
- Modify: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Modify: `docs/PARAMETERS.md`

- [x] **Step 1: Update public architecture and parameter documentation**
- [x] **Step 2: Run focused unit suites for every changed boundary**
- [x] **Step 3: Run the complete non-GPU unit suite**

  ```bash
  PYTHONPATH=. pytest tests/unit_tests -q
  ```

- [x] **Step 4: Run config composition and import validation**

  ```bash
  bash scripts/install/60_verify.sh
  ```

- [x] **Step 5: Run gated tiny Ray/GPU smoke tests when hardware is available**
- [x] **Step 6: Run `git diff --check` and inspect all changed files**
- [x] **Step 7: Remove fixed `[256,4096]` assumptions from active geometry boundaries**

  Derive policy geometry from the loaded VLA backbone, validate it against task
  metadata, and propagate the metadata through collection, sidecars, WM, and CLS.
  Keep `[256,4096]` only as the current checkpoint value and the narrow legacy
  sidecar-migration default.
