# PERF-H4 — gate per-step grad-norm / cosine diagnostics behind a debug flag

## Problem (audit H4 §1.2 / §3.3, pattern §B)
`imagine_actor_critic_step` in `dreamervla/algorithms/dreamervla.py` pays a large
diagnostic cost on **every** actor update:

- `_flat_grad` (≈ lines 1009–1066) runs up to **4 extra**
  `torch.autograd.grad(loss_t, actor_params, retain_graph=True, ...)` calls — one
  per loss component (PG, BC-ref, entropy, BC-vla) — purely to record per-component
  gradient L2 norms and the PG↔BC-ref cosine. `retain_graph=True` pins the whole
  actor activation graph for the duration of those extra grads.
- `_named_grad_norm` (≈ lines 1067–1073) does **5 independent full-parameter-name
  traversals** of `policy.named_parameters()` after the main backward, again
  diagnostic-only (per-submodule grad norms).

These are emitted as metrics (`actor_grad_norm_pg`, `..._bc_ref`, `..._entropy`,
`..._bc_vla`, `actor_grad_cos_pg_bcref`, `actor_grad_norm_{adapter,action_head,
output_projection,policy_head,log_std}`) but cost real compute + memory every step.

## retain_graph analysis (the load-bearing correctness point)
All `retain_graph=True` in this function lives **only** inside `_flat_grad`'s
`autograd.grad` (line 1020). The graph consumers are, in order:

1. `_flat_grad(actor_pg_loss)` → `autograd.grad(..., retain_graph=True)`
2. `_flat_grad(actor_bc_ref_loss_scaled)` (guarded) → same
3. `_flat_grad(actor_entropy_loss)` (guarded) → same
4. `_flat_grad(actor_bc_loss_scaled)` (guarded) → same
5. `actor_loss.backward()` (line 1066) — **default `retain_graph=False`**

So the graph is retained across the extra grads (1–4) so that each subsequent grad
call *and* the final `backward()` still have the graph. The **main backward never
requests `retain_graph`** — it frees the graph normally. The activation pinning is a
side-effect of the diagnostic grads running first, not of the main update.

Therefore: when the diagnostics are OFF and we skip steps 1–4, `actor_loss.backward()`
(step 5) runs exactly as today and frees the graph at the main backward. **No
`retain_graph` flag needs to be removed from the main backward — there is none.** The
memory/compute win is simply from not running the 4 extra grads + not retaining the
graph across them.

The critic backward (line 1212) is on a detached, independent graph and is untouched.

## Exact gating
Add an opt-in knob `optim_cfg.get("grad_diagnostics", False)` (default **OFF**),
matching the existing `optim_cfg.get("grad_clip_norm", ...)` /
`optim_cfg.get("zero_grad_set_to_none", ...)` pattern (callers already pass
`optim_cfg=cfg.optim`; no new threading). No existing per-step `debug` / `log_every`
knob is threaded into the algorithm, so a clearly-named opt-in is the minimal change.

When `grad_diagnostics` is **OFF** (default):
- Skip the four `_flat_grad` calls and the cosine computation → no extra backward,
  no `retain_graph`. The gated metrics default to `0.0` (matching the existing
  "unavailable → 0.0" convention for `_norm(None)` and the `cos_pg_bcref` else-branch).
- Skip the five `_named_grad_norm` calls → those metrics default to `0.0`.
- The main path is unchanged: `zero_grad`, `actor_loss.backward()`, `clip_grad_norm_`,
  `actor_optimizer.step()` all run identically.

When **ON**: behavior is byte-for-byte identical to today (same grads, same metrics).

`log_prob_mean`, `log_prob_std`, `advantage_pos_frac` are cheap (no extra backward),
diagnostic but not the H4 cost — out of scope, left always-on.

## Equivalence note (math must be identical when OFF)
The optimizer math (params after the step) depends only on `actor_loss`, the
`zero_grad`, the single `actor_loss.backward()`, `clip_grad_norm_`, and
`actor_optimizer.step()`. The gated code (`_flat_grad` extra grads, cosine,
`_named_grad_norm`) reads `.grad`/computes norms but **never mutates params, grads
that feed the step, the loss, or the optimizer state**. The extra `autograd.grad`
calls use `create_graph=False` and their results are discarded. Hence with the flag
OFF the post-step params are **identical** (atol=0) to a step that never computed the
diagnostics — we remove diagnostic-only work, not training math.

## TDD
CPU-only, tiny modules, in `tests/unit_tests/test_grad_diagnostics_gate.py`:

1. **RED driver — call-count**: monkeypatch/spy `_flat_grad` and `_named_grad_norm`
   (module-level for `_named_grad_norm`; `_flat_grad` is a closure, so spy via
   `torch.autograd.grad` call count which is the observable proxy for the extra
   backward). With `grad_diagnostics=False`: assert 0 extra `autograd.grad` calls and
   0 `_named_grad_norm` calls, and that params still updated (changed from init).
   Before the gate exists this RED-fails (diagnostics always run).
2. **Equivalence**: run one OFF-step and one reference step (same seed / same fresh
   modules + optimizer) and assert every `policy.parameter()` is equal with `atol=0`.
3. (Sanity, ON path) with `grad_diagnostics=True`: the gated metrics are populated /
   the extra `autograd.grad` calls happen — confirms ON is unchanged.

RED → GREEN, then `ruff check`.

## Scope
`dreamervla/algorithms/dreamervla.py` + the new test + this doc. No roadmap edits, no
config-file edits (knob is read with a default), no unrelated code.
