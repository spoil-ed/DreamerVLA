# PERF-W6 — micro-batch the dense / dense-chunk PPO update (bound RL-update GPU memory)

## Problem (audit W6 / roadmap §A — OOM surface)
`dino_wmpo_dense_step` (`dreamervla/algorithms/ppo/dense.py`) and
`dino_wmpo_dense_chunk_step` (`dreamervla/algorithms/ppo/dense_chunk.py`) each run a
**single backward over the whole imagined trajectory batch**. After imagining `horizon`
steps for the full effective batch `B_eff = B * group_size`, both files:

1. re-evaluate every imagined `(actor_feat, action)` through the policy (`mode="evaluate"`),
   building one autograd graph that holds **all `horizon` policy forwards for all `B_eff`
   rollouts at once**;
2. stack the per-step log-probs / entropies into `[B_eff, horizon]`;
3. reduce to a scalar actor loss and call `actor_loss.backward()` **once**.

The peak activation memory of that single backward scales with `B_eff * horizon`, which is
the RL-update OOM surface W6 targets (the WMPO outcome route already fixed its analogue under
MEM-RL-01 / `2026-06-22-mem-rl-01-microbatch-wmpo.md`). Reducing `B_eff` directly is exactly
the knob we want to avoid because it changes the GRPO statistics.

Audit line refs (verified against current main `4aa4346`):
- `dense.py`: epoch-0 actor backward at `dense.py:388-396`; the multi-epoch loop's backward at
  `dense.py:696-706`; the per-step `evaluate` forwards that build the graph at
  `dense.py:223-235` (epoch 0) and `dense.py:588-600` (later epochs).
- `dense_chunk.py`: the whole PPO update loop at `dense_chunk.py:290-332`; the single backward
  at `dense_chunk.py:327-332`; the per-step `evaluate` forwards at `dense_chunk.py:293-308`.

## The outcome.py pattern being reused (do NOT reinvent)
`dreamervla/algorithms/ppo/outcome.py` already implements the group-aligned micro-batched
backward this plan ports:

- A config knob `wmpo.update_micro_batch_starts` (`outcome.py:421-425`). It is read in *start*
  units (one start = `group_size` rollouts); `<= 0` or `>= n_starts` ⇒ ONE full-batch slice =
  bit-for-bit the original behavior. `slice_bounds` is a list of `(s_lo, s_hi)` start ranges.
- The B_eff axis is split into **contiguous, whole-group** slices `[s_lo*g : s_hi*g]`. GRPO
  groups are contiguous `group_size` blocks (`_repeat_latent` = `repeat_interleave`), and the
  advantage is computed ONCE on the full batch and **detached** before the loss, so slicing the
  detached `advantages[lo:hi]` reproduces the exact per-rollout weights — group alignment is
  only needed because the advantage is group-relative; once detached it is a per-rollout
  constant.
- Per slice → per chunk: forward `evaluate`, compute the per-rollout PPO term, normalize with
  `masked_mean_ratio_chunk_term(value, mask, count, B_eff)` (`grpo.py:49-67`) which divides by
  the **GLOBAL** `B_eff` (not the slice size), then `loss_c.backward()` and accumulate into the
  same `.grad` buffers (`outcome.py:639-748`). One `optimizer.step()` after the slice/chunk
  loop per epoch.

The numerical identity is gradient accumulation: `∂/∂θ Σ_slices (slice_sum / B_eff) =
(1/B_eff) Σ_all term = ∂/∂θ mean_over_B_eff(term)`. Summing per-slice gradients = full-batch
gradient, provided every term normalizes by the global `B_eff` and every per-rollout weight
(advantage) is the global one.

We reuse `_slice_latent` (`grpo.py:26-38`) — but note dense's stored per-step tensors
(`actor_feats[i]`, `actions[i]`, `old_log_probs[i]`) are plain `[B_eff, ...]` tensors, so a
simple `tensor[lo:hi]` slice is the analogue; `_slice_latent` is the same operation for the
dict/tensor latent. The latents are already detached host-side data after the rollout phase, so
slicing them is free.

## The axis we micro-batch and WHY it is equivalence-preserving
**Axis = the `B_eff` (rollout) axis**, split into contiguous whole-group slices, identical to
outcome.py. The dense actor loss is, term by term:

- **PPO clip term** (`dense.py:360-362`, `dense_chunk.py:317-319`):
  `_ppo_clip_term(ratio, advantages, …).mean()`. `ratio` is `[B_eff]` (a per-rollout trajectory
  ratio = `exp(Σ_h logp - Σ_h old_logp)`), `advantages` is `[B_eff]` **detached**. `.mean()` is
  a plain mean over `B_eff`. → exactly `mean_over_B_eff(per_rollout_term)`.
- **Entropy term** (`dense.py:363-367`, `dense_chunk.py:320-324`):
  `-(entropy_coef * entropy_stack.sum(dim=1)).mean()`. `entropy_stack.sum(dim=1)` is `[B_eff]`,
  `.mean()` over `B_eff`. → plain mean over `B_eff`.

Both are `mean_over_B_eff(f(rollout_i))` where `f(rollout_i)` depends ONLY on rollout `i`'s
stored data and its detached advantage. Slicing `B_eff` and normalizing each slice's
contribution by the **global** `B_eff` therefore makes the accumulated gradient bit-for-bit the
full-batch gradient (float-rounding aside).

Implementation of the per-slice mean-with-global-normalizer: instead of `.mean()` (which divides
by the slice size), each slice computes `term_slice.sum() / B_eff` and backprops that; summed
over slices = `(1/B_eff) Σ_all = mean`.

### Terms that are NOT a clean mean over the imagined B_eff axis — handled explicitly
1. **`bc_ref_loss`** (`dense.py:368-372`, only when `actor_bc_to_ref_scale > 0`):
   `torch.stack(bc_ref_losses).mean()` where each list element is itself a scalar
   `(action_chunk - ref).square().mean()` over `(B_eff, K, A)`. This still reduces to
   `mean_over_horizon( mean_over_(B_eff,K,A) )`, i.e. a mean over `B_eff` after dividing by
   `horizon * K * A`. To keep it equivalent under B_eff slicing we accumulate, per slice,
   `Σ_horizon Σ_slice (action_chunk-ref)^2 / (B_eff_global * K * A * horizon)`. Because `K`, `A`,
   `horizon` are constant across slices this is a global-B_eff normalization exactly like the PPO
   term. **The plain `.mean()` over `(slice_B, K, A)` is NOT slice-invariant**, so we must use
   the explicit `sum / (B_eff_global * K * A)` form.
2. **`real_relabel_term`** (`dense.py:373-386`): operates on a SEPARATE `real_relabel_batch`,
   independent of the imagined `B_eff` axis, gated by `real_rollout_relabel.loss_scale > 0`. It
   does NOT scale with the slice. To keep accumulation equivalent it is added **once** (on the
   first slice only), unchanged. Its gradient is identical whether added inside one full backward
   or one micro-batch backward.
3. **TD-MPC critic side-update** (`dense.py:398-578`): a SEPARATE loss on a SEPARATE optimizer
   (`critic_optimizer`), already its own `backward()`/`step()` block, NOT part of the actor
   backward. **Untouched** by W6 — the actor micro-batch loop sits where the single actor
   backward is today; the critic block runs exactly as before.
4. **Drift / clip diagnostic tensors** (`dense.py:262-281`, `dense_chunk.py:216-232`): all
   `.detach()`ed metrics, not in the loss graph. Computed per slice and concatenated/averaged
   for reporting; values are slice-invariant because they are per-element means we re-derive
   from the same data.

Conclusion: every loss term IS batch-separable along `B_eff` once `bc_ref_loss` uses the
explicit global normalizer and `real_relabel_term` is added once. No max / running-statistic /
batch-norm term exists in either actor loss. → micro-batching is numerically equivalent.

## Knob (default preserves current behavior)
Add `update_micro_batch_starts` read from the same `algorithm_cfg.wmpo` block both files already
read (`dense.py:89-130` reads `algorithm_cfg.get(...)` and `wmpo` is read in dense_chunk at
`dense_chunk.py:104`; for dense.py there is no `wmpo` block read today, so read
`algorithm_cfg.get("wmpo", {}).get("update_micro_batch_starts", 0)` to match outcome.py's
namespace). Default `0` ⇒ `mb_starts = n_starts` ⇒ ONE slice ⇒ original single backward,
bit-for-bit.

`n_starts = B_eff // group_size`; `slice_bounds = [(s, min(s+mb, n_starts)) for s in range(0,
n_starts, mb)]` exactly as outcome.py.

## TDD steps
1. `git reset --hard 4aa4346`; confirm HEAD. *(done)*
2. Write `tests/unit_tests/test_dense_microbatch_equivalence.py` (below). Run → RED.
3. Implement the micro-batch knob + slice loop in `dense_chunk.py` (simpler: single update loop).
   Run test → dense_chunk cases GREEN.
4. Implement in `dense.py` (epoch-0 path + multi-epoch loop share a slice helper). Run test →
   all GREEN.
5. `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "dense or ppo or microbatch"`
   → GREEN (pre-existing failures, if any, distinguished against a clean `4aa4346`).
6. `conda run -n dreamervla ruff check dense.py dense_chunk.py test_dense_microbatch_equivalence.py`
   → clean.
7. ONE `--signoff` commit, conventional `perf(ppo): …`, no `===` / `/`. Do NOT push.

## Equivalence test design (`tests/unit_tests/test_dense_microbatch_equivalence.py`)
Modeled on `tests/unit_tests/test_wmpo_microbatch_equivalence.py`:

- Deterministic mock WM (`observe_sequence` / `actor_input` / `predict_next` for dense,
  `predict_next_chunk` for dense_chunk) + a deterministic state-reward head; no RNG.
- A `_DetPolicy` whose `sample` returns the mean action plus a **per-rollout, slice-invariant**
  offset keyed on `arange(b) % group_size` (so within-group advantage variance is non-zero and
  the gradient is non-trivial), and whose `evaluate` returns a differentiable log-prob in the
  learnable `action_value` param. Same construction as the WMPO template.
- `_run_dense_update(micro_batch_starts, …)` / `_run_dense_chunk_update(...)` run the step with
  the micro-batch knob OFF (`0`, one slice) vs ON (`1`, one start per slice), reading
  `policy.action_value.grad` after a single `lr=0` step (grad read directly) and the final
  param after a `lr>0` multi-epoch run.
- Assertions:
  - `g_full.abs() > 1e-6` — guard the fixture is not a vacuous (identically-zero) gate.
  - `torch.allclose(g_full, g_micro, atol=1e-6)` — gradient parity (CPU float32; tighten to
    float64 if the policy/WM mocks are float64-clean).
  - multi-epoch param parity `torch.allclose(p_full, p_micro, atol=1e-6)`.
  - One case with `actor_bc_to_ref_scale > 0` + a frozen `ref_policy` for dense.py, asserting BC
    micro-batch parity (this is the term that needs the explicit global normalizer).

## Test / ruff gate
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_dense_microbatch_equivalence.py -q`
- `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "dense or ppo or microbatch"`
- `conda run -n dreamervla ruff check dreamervla/algorithms/ppo/dense.py dreamervla/algorithms/ppo/dense_chunk.py tests/unit_tests/test_dense_microbatch_equivalence.py`

## Scope guard
Touch ONLY `dreamervla/algorithms/ppo/dense.py`, `dreamervla/algorithms/ppo/dense_chunk.py`,
the new test, and this plan doc. The TD-MPC critic block, the rollout phase, and all metric keys
stay byte-identical.
