# Async cotrain (RLinf-style) wired into the coldstart main solution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`. The async path needs an 8× H100 + Ray run to validate end-to-end; only static checks (config compose, launcher dry_run, unit tests, ruff) are possible without GPU.

**Goal:** Make the coldstart→warmup→cotrain "main solution" able to run the cotrain stage as an **async (overlapped rollout⟂training) Ray** loop with **RLinf-style off-policy correctness** (version tracking + staleness throttle + off-policy logprob interpolation), selectable with one launcher knob.

**Architecture:** Two stages.
- **Stage 1 (wiring):** add `cotrain_engine=sync|async` to the launcher. `async` runs (a) the existing sync **pipeline warmup-only** (WM+classifier → ckpts), then (b) the existing async **`OnlineCotrainRayRunner._run_loop_overlap`** online loop initialized from those ckpts. Bridge the warmup ckpt format → the ray runner's `init_ckpt` format.
- **Stage 2 (RLinf correctness):** add version tracking (rollout stamps the policy version on each trajectory), staleness throttling (block generation when too far ahead of the learner), and off-policy logprob interpolation (alpha) in the wmpo/PPO actor update. Modeled on RLinf `rlinf/algorithms/losses.py:66-86`, `async_huggingface_worker.py:86-143`, `async_ppo_embodied_runner.py:107-152`.

**Tech stack:** Ray WorkerGroup + PackedPlacementStrategy, torch, Hydra, the existing `OnlineCotrainRayRunner` overlap loop, the wmpo_outcome actor update.

---

## Current reality (from code map)

- Async overlap EXISTS: `dreamervla/runners/online_cotrain_ray_runner.py` `_run_loop_overlap` (~293-520) overlaps infer⟂env-step⟂learner via Ray futures. Workers: Replay/Env/Inference/Learner (`_build_components` 77-156). Learner multi-GPU via `_learner_placement` (packed/flexible). It already tracks a `policy_version` int + `weight_sync_every` (sync loop 183-184,177-179) but does NOT stamp versions on trajectories, throttle staleness, or do off-policy correction.
- The ray runner has NO offline warmup; it loads pre-warmed weights via `_load_init_ckpt("inference.init_ckpt")` / `("learner.init_ckpt")` (124,134; loader 546-573 + 706-744).
- The sync pipeline (`OnlineCotrainPipelineRunner`) does warmup → writes `wm_warmup.ckpt` `{"world_model": sd}` and `classifier_warmup.ckpt` `{"classifier": sd, "classifier_threshold": f}` (online_cotrain_pipeline_runner.py warmup 35-72, save 75-109), then the sync online loop.
- The launcher (`coldstart_warmup_cotrain.py` build 151-281, main run 388/454-486) hardcodes `experiment=online_cotrain_pipeline_oft_action_hidden` (sync) for cotrain in both modes.
- Ray async experiment: `configs/experiment/online_cotrain_ray_oft.yaml` → `configs/dreamervla/ray_online_cotrain_rynn_action_hidden.yaml` (init.warmup_ckpt_path at 151,166).

---

## STAGE 1 — wire async into the coldstart launcher

### Task 1.1: warmup→ray init checkpoint bridge (pure, unit-tested)

**Files:** Create `dreamervla/runners/cotrain_warmup_bridge.py`; Test `tests/unit_tests/test_cotrain_warmup_bridge.py`

The pipeline writes two files (`wm_warmup.ckpt`, `classifier_warmup.ckpt`) + the VLA init lives at `task.openvla_oft.ckpt_path`. The ray runner's `_load_init_ckpt` expects a single torch file with `{"state_dicts": {"world_model":..., "classifier":..., "policy":...}}` (per loader 706-724). Write a pure consolidation function:

```python
def consolidate_warmup_ckpt(wm_ckpt: dict, cls_ckpt: dict, policy_sd: dict | None) -> dict:
    """Merge pipeline warmup ckpts into the ray runner's unified init format."""
    state_dicts = {"world_model": wm_ckpt["world_model"], "classifier": cls_ckpt["classifier"]}
    if policy_sd is not None:
        state_dicts["policy"] = policy_sd
    out = {"state_dicts": state_dicts}
    if "classifier_threshold" in cls_ckpt:
        out["classifier_threshold"] = float(cls_ckpt["classifier_threshold"])
    return out
```

Plus a thin file-level wrapper `write_consolidated_init(wm_path, cls_path, out_path, policy_path=None)` that torch.loads/saves. Unit-test `consolidate_warmup_ckpt` with fake dicts (keys present, threshold carried, policy optional). **Confirm at impl time** the exact `_load_init_ckpt` expected schema (read online_cotrain_ray_runner.py:546-573,706-744) and match it.

### Task 1.2: ray cotrain experiment with offline-warmup data wired

**Files:** Create `configs/experiment/online_cotrain_ray_oft_action_hidden.yaml`

Compose the OFT structural config + the ray runner, pointing `init.warmup_ckpt_path` at the consolidated bridge output, `warmup_steps=0`, and the OFT task. Mirror `online_cotrain_ray_oft.yaml` but for the OFT action_hidden structural dims (reuse `task.openvla_oft.*`). **Confirm at impl time** the dreamervla ray group keys (`ray_online_cotrain_rynn_action_hidden.yaml`).

### Task 1.3: launcher `cotrain_engine` switch

**Files:** Modify `configs/scripts/coldstart_warmup_cotrain.yaml`, `dreamervla/launchers/coldstart_warmup_cotrain.py`; Test `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`

- Config: add `cotrain_engine: sync` (sync|async).
- `build_pipeline_plan`: when `cotrain_engine=="async"`, emit TWO cotrain subcommands on the plan:
  1. `cotrain_warmup_cmd`: `experiment=online_cotrain_pipeline_oft_action_hidden online_rollout.total_env_steps=0` (warmup only, writes ckpts) — under torchrun if ngpu>1.
  2. `cotrain_online_cmd`: `experiment=online_cotrain_ray_oft_action_hidden init.warmup_ckpt_path=<consolidated>` (ray async) — NOT under torchrun (Ray owns placement).
  Plus a consolidation step (call the bridge) between them.
  When `cotrain_engine=="sync"` (default), keep the single `cotrain_cmd` (unchanged).
- `main()`: run collect → (sync: cotrain) | (async: cotrain_warmup → consolidate → cotrain_online).
- Tests: dry_run async emits both subcommands; the online one targets the ray experiment + carries `init.warmup_ckpt_path`; sync unchanged.

### Task 1.4: tutorial

**Files:** Modify `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`

Document `cotrain_engine=async` (RLinf-style overlapped cotrain): warmup runs sync (DDP), online runs Ray async; note GPU/Ray requirements.

### Stage 1 verification (no GPU)
- `pytest` the bridge + launcher tests; ruff; launcher `dry_run cotrain_engine=async` shows the two subcommands; Hydra `--cfg job` for the new ray experiment resolves.

---

## STAGE 2 — RLinf-style off-policy correctness

RLinf refs: version stamping `huggingface_worker.py:420-424`; staleness `async_huggingface_worker.py:86-103`; off-policy alpha `losses.py:66-86`; runner loop `async_ppo_embodied_runner.py:107-152`.

### Task 2.1: off-policy logprob interpolation (pure, unit-tested)

**Files:** add to the wmpo loss module (find via `dino_wmpo_outcome_step`); Test new unit test.

Pure function modeled on RLinf:
```python
def proximal_logprobs(old_lp, cur_lp, behav_v, theta_v):
    # alpha interpolates old->cur by how stale the behavior policy is
    denom = max(theta_v - behav_v, 1e-8)
    alpha = min(max((theta_v - 1 - behav_v) / denom, 0.0), 1.0)
    return old_lp + alpha * (cur_lp - old_lp)
```
Unit-test: behav_v==theta_v → alpha clamps (on-policy → old); larger gap → alpha→1 (use current). Wire into the wmpo actor update's ratio computation (guarded by a flag `algorithm.async_offpolicy.enabled`, default off → numerics unchanged for sync). **Confirm at impl time** the exact ratio site in `dino_wmpo_outcome_step`.

### Task 2.2: version stamping on trajectories

**Files:** rollout/inference worker (`rollout_inference_worker.py` / `inference_worker.py`) + replay record.

Stamp the current policy `version` on every transition the inference worker emits (mirror the existing `policy_version`/`local_infer_version` in the ray runner). Carry it through the replay so the learner can read `behav_v` per sample. **Confirm at impl time** the record schema + how the learner samples.

### Task 2.3: staleness throttle (pure, unit-tested)

**Files:** ray runner overlap loop.

Before launching a new rollout/inference, block if in-flight generations exceed `(staleness_threshold + learner_version + 1) * num_envs`. Pure predicate `should_throttle(generated, version, staleness_threshold, num_envs, ...) -> bool`, unit-tested. Knob `algorithm.staleness_threshold` (default 1, null=disable). Wire into `_run_loop_overlap` launch-gating.

### Task 2.4: weight-sync version increment

**Files:** ray runner overlap loop + learner/inference weight sync.

On each learner update + weight broadcast to inference, increment `learner_version`; inference stamps that version on subsequent rollouts. Add `weight_sync_interval` (default 1). Mostly EXISTS (policy_version/weight_sync_every) — extend to feed Task 2.1/2.3.

### Task 2.5: config knobs + experiment

**Files:** the ray dreamervla group + the new ray experiment.

Add `algorithm.staleness_threshold: 1`, `algorithm.async_offpolicy.enabled: true`, `actor.sync_weight_no_wait: false`, `runner.weight_sync_interval: 1`. Document defaults.

### Stage 2 verification
- Unit tests for proximal_logprobs, should_throttle, version carry.
- GPU (user, elsewhere): run async cotrain, confirm overlap_events>0, training progresses, off-policy ratio finite, no divergence vs sync.

---

## NOT covered / risks
- Offline warmup still runs SYNC (single/DDP) before the async online phase; warmup itself is not async.
- Multi-GPU async learner is via Ray placement (packed/flexible); FSDP under Ray not wired.
- The async overlap loop is under-tested (no overlap-metric assertions today); add at least one.
- Full correctness of off-policy wmpo needs GPU validation; keep `async_offpolicy.enabled` default-off until validated so sync numerics are untouched.
- Generation backend: DreamerVLA uses HF/torch (OFT), NOT SGLang/vLLM — skip RLinf's vLLM weight-syncer specifics.
