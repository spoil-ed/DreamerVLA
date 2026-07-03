# RLinf-WoVR Full-Pipeline Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the training-side (non-inference) alignment gaps between DreamerVLA's manual Ray cotrain pipeline and the RLinf-WoVR reference, wiring already-built mechanisms into the *activated* route and adding calibration/overlap that RLinf has and we lack.

**Architecture:** The mainline `experiment=openvla_onetraj_libero_cotrain_ray` drives the actor update through `EmbodiedFSDPActor.run_training` (the "activated route"); the richer `dreamervla/algorithms/ppo/outcome.py` route (with variance filtering, per-rollout equal-weight normalization, and group-aligned micro-batching) is **not active** under this config. Three of those mechanisms are ported *into* the activated actor (Sub-plan A). Sub-plan B calibrates the success threshold during warmup (currently a hard `0.5` config default). Sub-plan C adds env-bootstrap overlap and deterministic eval enumeration.

**Tech Stack:** Python 3.11, PyTorch, FSDP1/2, Ray (single-node), Hydra/OmegaConf, pytest. Run tests in the **`dreamervla`** conda env (clean baseline ≈ 1328 passed).

## Global Constraints

- Run unit tests in the `dreamervla` conda env: `conda run -n dreamervla pytest ...` (base env gives ~13 spurious failures).
- Commits require `--signoff`; subjects reject `===` and `/`; end body with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Numerics-changing defaults must be switchable back to the original behavior via config knob (project convention: "switchable-default-original"). New knobs default to the value that **reproduces current behavior** unless a task explicitly flips the default and says why.
- `outcome.py` lives at `dreamervla/algorithms/ppo/outcome.py` (NOT `algorithms/imagine/`). The grpo helper for per-rollout equal weight is `masked_mean_ratio_chunk_term` in `dreamervla/algorithms/ppo/grpo.py` (NOT `masked_mean_ratio`).
- Do not add new top-level route YAMLs; extend the existing cotrain config + `actor.train_cfg.algorithm_cfg`.
- "WoVR" / removed RL-route wording is forbidden in active source files (enforced by `test_repository_hygiene.py`); this plan doc and `docs/rlinf_wovr_inference_optimizations.md` are the sanctioned exceptions.

---

## Execution Status (2026-07-03, branch `feat/rlinf-alignment-full-pipeline`)

- ✅ **A1** committed `01ba4f0` — `grpo.group_variance_mask` + activated-actor wiring; tests green.
- ✅ **A2** committed `00d11f7` — per-rollout equal-weight loss normalization; tests green.
- ✅ **A4** committed `c9ee022` — enabled A1+A2 knobs in the mainline cotrain config; compose test green.
- ⏸ **A3** (group-aligned micro-batch) — DEFERRED. The activated `run_training` loops over *time steps* evaluating the full rollout batch per step (`embodied_fsdp_actor.py:231-315`); micro-batching the rollout dim requires slicing `_eval_inputs_for_step` output per group-aligned slice + an exact full-vs-micro equivalence test. Not landed to avoid a wrong numerics-changing edit under time pressure. Next task.
- ⏳ **B1/B2** (threshold calibration + warmup val gate) — not started; anchors verified in plan.
- ⏳ **C1/C2** (bootstrap overlap + eval enumeration) — not started; both need the `REQUIRED READ` first steps (env-worker `interact`, eval loop) that the extraction agents could not finish (session limit reset 1:50pm ET).

Regression at checkpoint: `test_embodied_fsdp_actor + test_grpo_helpers + test_manual_cotrain_ray_runner + test_manual_cotrain_config_validation` = **109 passed**.

## File Structure

| File | Responsibility | Sub-plan |
|---|---|---|
| `dreamervla/workers/actor/embodied_fsdp_actor.py` | Activated actor: advantages + PPO update. Add variance filter, per-rollout weight, micro-batch. | A |
| `dreamervla/algorithms/ppo/grpo.py` | Shared GRPO helpers. Add a group-variance-mask helper reused by the actor. | A |
| `tests/unit_tests/test_embodied_fsdp_actor.py` | Actor unit tests (fixtures `_actor_cfg`/`_shard`). | A |
| `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml` | `actor.train_cfg.algorithm_cfg` knobs. | A |
| `dreamervla/runners/online_cotrain_pipeline_runner.py` | Warmup orchestration: add held-out F1 threshold calibration + val gate. | B |
| `dreamervla/runners/latent_classifier_runner.py` | Owns `_sweep_metrics` (reused by B). | B |
| `dreamervla/runners/manual_cotrain_ray_runner.py` | `_run_global_step`: add bootstrap overlap. | C |
| `dreamervla/workers/env/trajectory_env_worker.py` | Env worker: add `prefetch_bootstrap`/consume path. | C |
| `dreamervla/runners/embodied_eval_runner.py` | Eval loop: deterministic ordered enumeration. | C |
| `configs/evaluation/libero_vla.yaml` | Eval enumeration knob. | C |

---

# Sub-plan A — Actor RL signal quality (activated route)

Highest leverage: all three mechanisms already exist in `outcome.py`; this wires their *semantics* into `EmbodiedFSDPActor`. Each defaults to reproducing current behavior; the config flips them on for the mainline.

**Ground-truth current code (do not assume — verified 2026-07-03):**

`embodied_fsdp_actor.py:141-175` `compute_advantages_and_returns` computes `returns = _trajectory_returns_from_rewards(...)` then `advantages = _group_advantage(returns, group_size, eps=1e-6)` with NO variance filtering.

`embodied_fsdp_actor.py:206-303` `run_training` normalizes loss by a single global `valid_count = _distributed_sum_int(loss_mask.sum())`: `loss = (ppo_clip.sum() - entropy_coef*entropy.sum()) / float(valid_count)`; there is NO mini/micro-batch split (whole time-batch accumulated before one `optimizer.step()`).

`grpo.py:110-138` `_group_advantage` does group z-score. `grpo.py:49-67` `masked_mean_ratio_chunk_term(value_vec, mask_c, per_rollout_count, b_eff)` returns `((value_vec*mask_c)/per_rollout_count).sum()/float(b_eff)`.

---

### Task A1: Group-variance filter in the activated actor

**Files:**
- Modify: `dreamervla/algorithms/ppo/grpo.py` (add `group_variance_mask`)
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py:141-175` (`compute_advantages_and_returns`)
- Test: `tests/unit_tests/test_embodied_fsdp_actor.py`, `tests/unit_tests/test_grpo_helpers.py` (create if absent)

**Interfaces:**
- Produces: `grpo.group_variance_mask(returns: torch.Tensor, group_size: int, eps: float) -> torch.Tensor` — returns a `[N]` float32 mask, `0.0` for every rollout whose group has `std <= eps` (all-success/all-fail), else `1.0`. Mirrors `outcome.py::_adaptive_group_advantage_and_mask`'s keep/skip decision but for the fixed-width (non-adaptive) actor batch.
- Consumes (actor): after computing `advantages`, multiply into `self.advantages` AND record a mask that `run_training` applies to `loss_mask`. Add `self.group_variance_mask: torch.Tensor | None`.

- [ ] **Step 1: Write the failing test for the helper**

```python
# tests/unit_tests/test_grpo_helpers.py
import torch
from dreamervla.algorithms.ppo.grpo import group_variance_mask


def test_group_variance_mask_zeros_degenerate_groups():
    # group 0 = all-success (no variance), group 1 = mixed (has variance)
    returns = torch.tensor([1.0, 1.0, 0.0, 1.0])
    mask = group_variance_mask(returns, group_size=2, eps=1e-6)
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_group_variance_mask_all_kept_when_group_size_one():
    returns = torch.tensor([1.0, 0.0, 1.0])
    mask = group_variance_mask(returns, group_size=1, eps=1e-6)
    assert mask.tolist() == [1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_grpo_helpers.py -q`
Expected: FAIL with `ImportError: cannot import name 'group_variance_mask'`.

- [ ] **Step 3: Implement the helper in `grpo.py`** (add after `_group_advantage`)

```python
def group_variance_mask(
    score: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """Return a [N] float mask zeroing rollouts whose GRPO group has no return
    variance (all-success or all-fail). Their group-relative advantage is 0
    anyway, so masking them out of the loss is a pure compute/stability win.
    Mirrors ``outcome.py::_adaptive_group_advantage_and_mask`` for the
    fixed-width actor batch; ``group_size <= 1`` keeps everything."""
    g = int(group_size)
    n = int(score.numel())
    if g <= 1:
        return torch.ones_like(score)
    if n < g or n % g != 0:
        raise ValueError(
            f"group_variance_mask: numel={n} not a positive multiple of group_size={g}"
        )
    groups = score.reshape(-1, g)
    has_var = groups.std(dim=1, unbiased=False) > float(eps)  # [n_groups]
    return has_var.float().repeat_interleave(g).reshape_as(score)
```

- [ ] **Step 4: Run helper test to verify it passes**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_grpo_helpers.py -q`
Expected: PASS.

- [ ] **Step 5: Write failing actor test**

```python
# tests/unit_tests/test_embodied_fsdp_actor.py  (add)
def test_actor_masks_zero_variance_groups_when_enabled() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["algorithm_cfg"]["filter_zero_variance_groups"] = True
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    # group_size=2: two all-fail rollouts (no variance) -> masked out
    actor.load_trajectory_shards([_shard(0.0, 0.0)])
    metrics = actor.compute_advantages_and_returns()
    assert metrics["actor/zero_variance_masked_rollouts"] == 2.0
```

- [ ] **Step 6: Run to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_masks_zero_variance_groups_when_enabled -q`
Expected: FAIL (`KeyError: 'actor/zero_variance_masked_rollouts'`).

- [ ] **Step 7: Wire into `compute_advantages_and_returns`** (after the `advantages = _group_advantage(...)` line)

```python
        advantages = _group_advantage(returns, group_size, eps=1e-6)
        if bool(algorithm_cfg.get("filter_zero_variance_groups", False)):
            var_mask = _group_variance_mask(returns, group_size, eps=1e-6)
        else:
            var_mask = torch.ones_like(returns)
        self.group_variance_mask = var_mask.detach()
        self.returns = returns.detach()
        self.advantages = (advantages * var_mask).detach()
```

Add to `self._advantage_metrics`:
```python
            "actor/zero_variance_masked_rollouts": float(
                (var_mask <= 0.0).sum().cpu().item()
            ),
```

Add the import binding next to the other grpo helpers (`embodied_fsdp_actor.py:610-614`):
```python
_group_variance_mask = _GRPO_HELPERS.group_variance_mask
```
And initialize `self.group_variance_mask: torch.Tensor | None = None` in `__init__`.

- [ ] **Step 8: Apply the mask to `loss_mask` in `run_training`**

In `run_training`, after `loss_mask` is materialized and before `valid_count` is computed, gate columns by the per-rollout variance mask so masked rollouts contribute zero loss AND are excluded from `valid_count`:
```python
        if self.group_variance_mask is not None:
            loss_mask = loss_mask * self.group_variance_mask.to(
                loss_mask.device, dtype=loss_mask.dtype
            ).reshape((1, -1) + (1,) * (loss_mask.ndim - 2))
```
(`loss_mask` is `[time, batch, ...]`; the reshape broadcasts the per-rollout `[batch]` mask over time.)

- [ ] **Step 9: Run actor + full-suite-adjacent tests**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py tests/unit_tests/test_grpo_helpers.py -q`
Expected: PASS (existing tests unchanged because the knob defaults to `False`).

- [ ] **Step 10: Commit**

```bash
git add dreamervla/algorithms/ppo/grpo.py dreamervla/workers/actor/embodied_fsdp_actor.py tests/unit_tests/test_grpo_helpers.py tests/unit_tests/test_embodied_fsdp_actor.py
git commit --signoff -m "feat: filter zero-variance GRPO groups in activated actor"
```

---

### Task A2: Per-rollout equal-weight loss normalization

Current activated normalization divides the summed clip loss by a single global `valid_count`, over-weighting long/failed rollouts. RLinf's `masked_mean_ratio` weights every rollout equally regardless of length. Add an opt-in that reproduces the per-rollout-equal-weight semantics; default `False` keeps bit-exact current behavior.

**Files:**
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py` (`run_training` loss aggregation)
- Test: `tests/unit_tests/test_embodied_fsdp_actor.py`

**Interfaces:**
- Consumes: `algorithm_cfg["loss_normalization"] in {"global_valid_count" (default), "per_rollout"}`.
- For `per_rollout`: weight each PPO term by `1 / (per_rollout_valid_count * num_rollouts)` instead of `1 / valid_count`, where `per_rollout_valid_count[b] = loss_mask[:, b].sum()` clamped `>= 1`.

- [ ] **Step 1: Write failing test** — two shards of different valid length with equal advantage magnitude produce equal per-rollout gradient weight.

```python
def test_actor_per_rollout_normalization_equalizes_rollout_weight() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["algorithm_cfg"]["loss_normalization"] = "per_rollout"
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    actor.load_trajectory_shards(
        [_variable_length_shard(steps=1, slot_id=0, reward=0.0),
         _variable_length_shard(steps=3, slot_id=1, reward=1.0)]
    )
    actor.compute_advantages_and_returns()
    metrics = actor.run_training()
    assert metrics["actor/loss_normalization_per_rollout"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_per_rollout_normalization_equalizes_rollout_weight -q`
Expected: FAIL (`KeyError`).

- [ ] **Step 3: Implement** — compute per-rollout counts once before the epoch loop:

```python
        loss_norm = str(algorithm_cfg.get("loss_normalization", "global_valid_count"))
        if loss_norm == "per_rollout":
            # [batch] valid step count per rollout, all-reduced batch dim is local
            per_rollout_count = loss_mask.reshape(loss_mask.shape[0], loss_mask.shape[1], -1).sum(dim=(0, 2)).clamp(min=1.0)
            num_rollouts = _distributed_sum_int(int(loss_mask.shape[1]), self.torch_device)
```

Replace `loss = (ppo_clip.sum() - entropy_coef*entropy.sum()) / float(valid_count)` with a branch: for `per_rollout`, weight the per-(time,batch) `ppo_clip`/`entropy` element-wise by `1/per_rollout_count[b]` then divide by `num_rollouts`. Record `metrics["actor/loss_normalization_per_rollout"] = 1.0 if loss_norm == "per_rollout" else 0.0`.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py -q`
Expected: PASS (default branch unchanged).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/actor/embodied_fsdp_actor.py tests/unit_tests/test_embodied_fsdp_actor.py
git commit --signoff -m "feat: add per-rollout equal-weight loss normalization to actor"
```

---

### Task A3: Group-aligned micro-batch in the activated actor

Port `outcome.py`'s `update_micro_batch_starts` semantics (contiguous whole-group slices) into `run_training`, bounding peak memory (known `full -> OOM@82GB`). Knob `update_micro_batch_starts <= 0` reproduces the single full-batch pass bit-for-bit.

**Files:**
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py` (`run_training` batch loop)
- Test: `tests/unit_tests/test_embodied_fsdp_actor.py`

**Interfaces:**
- Consumes: `algorithm_cfg["update_micro_batch_starts"] : int` (default `0`). Slices the batch dim into contiguous blocks of `mb_starts * group_size` rollouts. Grad-accumulate across slices; single `optimizer.step()` per epoch. Loss denominator stays the GLOBAL `valid_count` (or per-rollout `num_rollouts`) so the accumulated gradient equals the full-batch one.

- [ ] **Step 1: Write failing equivalence test** — micro-batched update produces (within `atol=1e-5`) the same post-step parameters as the full-batch update.

```python
def test_actor_microbatch_matches_full_batch_update() -> None:
    def run(mb_starts):
        cfg = _actor_cfg()
        cfg["train_cfg"]["algorithm_cfg"]["update_micro_batch_starts"] = mb_starts
        actor = EmbodiedFSDPActor(**cfg)
        actor.init(); torch.manual_seed(0)
        actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(1.0, 0.0)])
        actor.compute_advantages_and_returns(); actor.run_training()
        return [p.detach().clone() for p in actor.policy.parameters()]
    full = run(0)
    micro = run(1)
    for a, b in zip(full, micro):
        assert torch.allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_microbatch_matches_full_batch_update -q`
Expected: FAIL (knob ignored → currently passes only trivially; assert it fails because slicing not implemented — if it passes at mb=1 by luck, tighten with mb_starts spanning a partial group to force the code path).

- [ ] **Step 3: Implement group-aligned slicing** — mirror `outcome.py:634-652`:

```python
        group_size = int(algorithm_cfg.get("group_size", 1))
        n_starts = int(loss_mask.shape[1]) // group_size
        mb_cfg = int(algorithm_cfg.get("update_micro_batch_starts", 0))
        mb_starts = n_starts if mb_cfg <= 0 else min(max(1, mb_cfg), n_starts)
        slice_bounds = [(s * group_size, min(s + mb_starts, n_starts) * group_size)
                        for s in range(0, n_starts, mb_starts)]
```

In the epoch loop, `zero_grad` once, iterate `slice_bounds`, slice `advantage`/`old_logprob`/inputs on the batch dim `[lo:hi]`, accumulate `loss.backward()` per slice (loss still divided by the global denominator), then a single `grad_norm`/`optimizer.step()` after all slices.

- [ ] **Step 4: Run equivalence + suite**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_embodied_fsdp_actor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/actor/embodied_fsdp_actor.py tests/unit_tests/test_embodied_fsdp_actor.py
git commit --signoff -m "feat: group-aligned micro-batch in activated actor update"
```

---

### Task A4: Enable the knobs in the mainline config

**Files:**
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml` (`actor.train_cfg.algorithm_cfg`)
- Test: `tests/unit_tests/test_manual_cotrain_config_validation.py`

- [ ] **Step 1: Add a config-validation test** asserting the three knobs resolve.

```python
def test_cotrain_actor_enables_rl_signal_knobs() -> None:
    cfg = _load_cotrain_cfg()  # existing helper in this test module
    ac = cfg.actor.train_cfg.algorithm_cfg
    assert bool(ac.filter_zero_variance_groups) is True
    assert str(ac.loss_normalization) == "per_rollout"
    assert int(ac.update_micro_batch_starts) >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_manual_cotrain_config_validation.py -k rl_signal -q`
Expected: FAIL.

- [ ] **Step 3: Add knobs to `algorithm_cfg`** (`openvla_onetraj_libero_cotrain_ray.yaml`, in the `actor.train_cfg.algorithm_cfg` block):

```yaml
      filter_zero_variance_groups: true
      loss_normalization: per_rollout
      update_micro_batch_starts: 1
```

- [ ] **Step 4: Run to verify pass + full config suite**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_manual_cotrain_config_validation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml tests/unit_tests/test_manual_cotrain_config_validation.py
git commit --signoff -m "feat: enable actor RL-signal knobs in mainline cotrain config"
```

---

# Sub-plan B — Warmup success-threshold calibration + val gate

**Ground truth:** `classifier_threshold` is read from `algorithm.lumos.classifier_threshold` default **0.5** (`online_cotrain_runner.py:325-326`) and plumbed into the WM env at `online_cotrain_runner.py:1167`. Warmup `_offline_warmup_classifier` (`online_cotrain_pipeline_runner.py:172-229`) trains only — it returns `last_acc` and never sweeps a threshold. The F1 sweep `_sweep_metrics(probs, ys, thresholds, tag)` exists only in `latent_classifier_runner.py:663-689` (returns `{"best_f1", "best_thresh", ...}`). The env uses `success_by_slot = (rewards >= self.success_threshold).any(axis=1)` (`latent_world_model_env.py:552`).

### Task B1: Calibrate `classifier_threshold` from a held-out F1 sweep at warmup end

**Files:**
- Modify: `dreamervla/runners/latent_classifier_runner.py` (export `_sweep_metrics` — move to a shared spot or re-export; it is currently module-private)
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py` (`_offline_warmup_classifier`)
- Test: `tests/unit_tests/test_online_cotrain_pipeline_runner.py` (or the existing pipeline warmup test module — confirm name via `grep -rl "_offline_warmup_classifier\|classifier_threshold" tests/unit_tests`)

**Interfaces:**
- Consumes: `_sweep_metrics(probs, ys, thresholds, tag) -> dict` (add to `__all__` in `latent_classifier_runner.py`).
- Produces: after warmup training, `_offline_warmup_classifier` gathers held-out `(probs, ys)` from a val split of `replay`, calls `_sweep_metrics`, and sets `self.classifier_threshold = best_thresh` (guarded by `algorithm.lumos.calibrate_threshold`, default `False` to preserve `0.5`).

- [ ] **Step 1: Export the sweep** — add `_sweep_metrics` to `__all__` in `latent_classifier_runner.py` and write a test asserting importability + best-threshold selection:

```python
# tests/unit_tests/test_classifier_threshold_sweep.py
import numpy as np
from dreamervla.runners.latent_classifier_runner import _sweep_metrics

def test_sweep_picks_separating_threshold():
    probs = np.array([0.1, 0.2, 0.8, 0.9]); ys = np.array([0, 0, 1, 1])
    out = _sweep_metrics(probs, ys, np.linspace(0.1, 0.9, 9), "val")
    assert out["best_f1"] == 1.0
    assert 0.2 < out["best_thresh"] <= 0.8
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_classifier_threshold_sweep.py -q`
Expected: FAIL (`ImportError` if `_sweep_metrics` not exported).

- [ ] **Step 3: Export + implement calibration** in `_offline_warmup_classifier`, guarded by a `calibrate: bool = False` param wired from `algorithm.lumos.calibrate_threshold`. After the training loop, if `calibrate`, collect held-out probs/labels (reuse the classifier's eval batching against a held-out slice of `replay`), call `_sweep_metrics`, log `eval/classifier_warmup_best_f1`/`eval/classifier_warmup_best_thresh`, and set `self.classifier_threshold = best_thresh`.

- [ ] **Step 4: Run pipeline warmup test + sweep test**

Run: `conda run -n dreamervla pytest tests/unit_tests/test_classifier_threshold_sweep.py <pipeline_warmup_test_module> -q`
Expected: PASS (default path unchanged when `calibrate_threshold=False`).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/latent_classifier_runner.py dreamervla/runners/online_cotrain_pipeline_runner.py tests/unit_tests/test_classifier_threshold_sweep.py
git commit --signoff -m "feat: calibrate classifier success threshold from held-out F1 at warmup"
```

### Task B2: Warmup held-out validation gate

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py` (after WM + classifier warmup, before returning to online)
- Test: same pipeline test module

**Interfaces:**
- Produces: `_warmup_val_gate(...) -> dict[str, float]` logging `eval/wm_warmup_val_loss`, `eval/classifier_warmup_val_f1`; raises `RuntimeError` if `classifier val F1 < algorithm.lumos.warmup_min_val_f1` (default `0.0` = gate disabled, preserving current behavior).

- [ ] **Step 1: Write failing test** — with `warmup_min_val_f1 = 0.9` and a classifier that scores below it, the gate raises.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement the gate** using the same held-out slice as B1.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat: add warmup held-out validation gate before online cotrain`.

---

# Sub-plan C — Throughput: bootstrap overlap + deterministic eval

### Task C1: env bootstrap overlap during actor training

**Ground truth:** `_run_global_step` (`manual_cotrain_ray_runner.py:279-467`) calls `real_env.interact(...)` (line 331), waits env+rollout (368-373), receives trajectories (388-396), then `actor.run_training()` (401). The next global step's `real_env.interact()` re-runs `bootstrap_obs`/`env.reset()` inline. RLinf overlaps this: `env.prefetch_train_bootstrap()` is launched during `actor.run_training()` so the next reset happens concurrently, and is force-disabled when offload is on (`config.py:901-903`).

**Files:**
- Modify: `dreamervla/workers/env/trajectory_env_worker.py` (add `prefetch_bootstrap()` that resets slots and caches first obs; `interact` consumes the cache if present)
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py:_run_global_step` (launch `real_env.prefetch_bootstrap()` right after `actor.run_training()` is dispatched, not `.wait()`-ed, so it overlaps)
- Test: `tests/unit_tests/test_trajectory_env_worker.py`, `tests/unit_tests/test_manual_cotrain_ray_runner.py`

**Interfaces:**
- Produces: `BaseTrajectoryEnvWorker.prefetch_bootstrap() -> dict[str, float]` — resets all slots, stores the first-obs batch in `self._prefetched_bootstrap`; idempotent. `interact(...)` uses `self._prefetched_bootstrap` if set (and clears it) instead of calling `bootstrap_obs` inline.
- Consumes (runner): a knob `manual_cotrain.overlap_env_bootstrap: bool` (default `False`). When true AND offload disabled, dispatch prefetch during training.

- [ ] **Step 1 (REQUIRED READ):** Read `trajectory_env_worker.py` around the `interact`/`bootstrap_obs`/`_step_slot` region (grep `def interact`, `bootstrap_obs`, `reset`) to capture the exact reset call and obs cache shape before writing the test. Record the real method names in this task before proceeding.
- [ ] **Step 2: Write failing worker test** — `prefetch_bootstrap()` populates a cache; a subsequent `interact` does NOT call `bootstrap_obs` again (assert via a call counter / monkeypatch).
- [ ] **Step 3: Run to verify it fails.**
- [ ] **Step 4: Implement `prefetch_bootstrap` + cache consumption in `interact`.**
- [ ] **Step 5: Run worker test — PASS.**
- [ ] **Step 6: Write failing runner test** — with `overlap_env_bootstrap=True`, `_run_global_step` dispatches `real_env.prefetch_bootstrap` after `actor.run_training` (assert ordering via a recording mock group).
- [ ] **Step 7: Implement runner wiring** with the offload guard (mirror `config.py:901-903`: if offload enabled, log once and skip overlap).
- [ ] **Step 8: Run runner test + full runner suite — PASS.**
- [ ] **Step 9: Commit** `feat: overlap env bootstrap reset with actor training`.

### Task C2: deterministic ordered eval enumeration

**Ground truth:** `configs/evaluation/libero_vla.yaml:53` uses `num_episodes_per_task: 3` with random sampling. RLinf enumerates `(task_id, trial_id)` deterministically (no shuffle when `is_eval`), covering all init states. Memory note: single-task 0% under random sampling is sampling bias, not garbage — deterministic enumeration removes that ambiguity.

**Files:**
- Modify: `dreamervla/runners/embodied_eval_runner.py` (episode loop)
- Modify: `configs/evaluation/libero_vla.yaml` (add `eval.enumerate_all_init_states: bool`)
- Test: `tests/unit_tests/test_embodied_eval_*` (confirm exact module via `grep -rl "num_episodes_per_task\|EmbodiedEvalRunner" tests/unit_tests`)

**Interfaces:**
- Consumes: `eval.enumerate_all_init_states: bool` (default `False` = current random `num_episodes_per_task` behavior). When `True`, episodes per task = the task's full init-state count, iterated in order (no RNG).

- [ ] **Step 1 (REQUIRED READ):** Read `embodied_eval_runner.py` around the task/episode loop (grep `num_episodes_per_task`, `success_once`, `reset`) to capture how init states are selected before writing the test.
- [ ] **Step 2: Write failing test** — with `enumerate_all_init_states=True`, the runner requests each init-state index exactly once, in order (assert against a mock env exposing N init states).
- [ ] **Step 3: Run to verify it fails.**
- [ ] **Step 4: Implement the ordered-enumeration branch.**
- [ ] **Step 5: Run to verify pass.**
- [ ] **Step 6: Add the config knob + a config test.**
- [ ] **Step 7: Commit** `feat: deterministic ordered eval init-state enumeration`.

---

## Design notes (scope decisions, verified)

- **Dense/relative reward registry entry is NOT on the activated path** — reprioritized out of this plan. The reward registry (`sparse_outcome`/`probability_outcome`) is consumed only by `outcome.py`, which is inactive under the mainline config. In the activated route the WM env already emits *dense per-step classifier probability* as reward (`latent_world_model_env.py:548`) and the actor sums it (`_trajectory_returns_from_rewards`). A true RLinf-style relative/differential reward would require changing the env's emitted reward or the actor's return computation — a larger, numerics-sensitive change deferred until the activated route's summed-probability return is shown to mis-behave (length/saturation bias, which Task A2's per-rollout normalization partially addresses).
- **Async cotrain + staleness backpressure: DEFERRED.** Highest port cost (resident workers + channel + decoupled-PPO importance correction); AGENTS.md/CLAUDE.md keep async behind explicit experiments. Revisit only after the synchronous path is confirmed throughput-bound.
- **`pipeline_stage_num` intra-worker pipelining: NOT PORTED.** Multi Ray-worker + the dynamic WM lease pool already cover its cross-worker overlap benefit; marginal gain in our env structure.
- **NCCL bucket weight sync, `model_weights_id` hash, rollout `torch.compile`/CUDA-graph:** optional polish, GPU-verify-gated, out of scope here.

---

## Self-Review

**Spec coverage vs the audit's ranked list:**
- A-group "influences convergence signal quality" items 1 (variance filter → A1), 3 (per-rollout equal weight → A2) ✅. Item 6 (micro-batch, throughput/scale) → A3 ✅. Config activation → A4 ✅.
- Threshold calibration (audit B1) → Task B1 ✅. Warmup val gate (audit B5) → Task B2 ✅.
- env bootstrap overlap (audit B7) → Task C1 ✅. Eval ordered enumeration (audit B8) → Task C2 ✅.
- Dense reward (audit B4) → reprioritized with justification (Design notes) — not silently dropped.
- Async / pipeline_stage_num / NCCL / compile → deferred with justification (Design notes).

**Placeholder scan:** Tasks C1/C2 contain explicit `REQUIRED READ` first steps because the eval loop and `trajectory_env_worker.interact` internals were not fully captured at plan time (two extraction agents hit a session limit). These are not silent TODOs — they are scoped, named reads with the exact grep targets. All Sub-plan A and B tasks carry complete code (verified against current source 2026-07-03).

**Type consistency:** helper is `group_variance_mask` everywhere; env attribute `success_threshold`; runner attribute `classifier_threshold`; grpo per-rollout helper `masked_mean_ratio_chunk_term`. Config knobs: `filter_zero_variance_groups`, `loss_normalization`, `update_micro_batch_starts`, `calibrate_threshold`, `warmup_min_val_f1`, `overlap_env_bootstrap`, `enumerate_all_init_states`.
