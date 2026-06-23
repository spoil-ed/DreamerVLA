# Perf Q2 + Q6 — drop redundant CPU clones, batch inference D2H

Date: 2026-06-23
Source audit: `docs/plans/performance_optimization_audit.md` (Q2 §I/§3.7, Q6 §A/§3.7, §4 constraints).
Style: follows `docs/plans/2026-06-22-mem-rl-01-microbatch-wmpo.md` — numerically identical,
memory/throughput-only, equivalence-as-safety-gate.

## Constraint recap (§4.4 of the audit)

A weight-sync state dict captured from `module.state_dict()` aliases the **live** parameter
storage. `param.detach()` keeps the alias. `param.detach().cpu()` then:

- source on **CUDA** → `.cpu()` ALWAYS allocates a fresh CPU tensor → already independent of the
  live param; a later optimizer `step()` mutates the CUDA storage, never this CPU copy. The
  trailing `.clone()` is a pure redundant full-model CPU copy → safe to drop.
- source on **CPU** → `.cpu()` is a no-op that returns the SAME storage (alias of the live param).
  An optimizer `step()` before the push would corrupt the captured tensor in place. The `.clone()`
  is load-bearing → must keep.

The helpers (`_to_cpu_tensor`, `_cpu_state_dict`, `compress_state_dict`) are **device-agnostic**:
the same function is reachable from a CUDA learner (the hot, OOM-relevant path) and from a CPU
model (tests, CPU-resident components). I cannot prove "always CUDA" at these call sites.

### Decision: device-conditional clone (provably correct everywhere)

Replace `value.detach().cpu().clone()` with: detach → `.cpu()`; clone ONLY if `.cpu()` did not
already produce a new tensor (i.e. the source was already on CPU). On CUDA this drops the
redundant copy; on CPU it preserves the §4-required independent copy. The returned tensor is an
independent CPU copy in BOTH cases, so no caller can observe a behavior change — only the GPU
fast path stops doing a second allocation. A tiny shared helper `_independent_cpu(t)` expresses
this once and is reused at every device-agnostic site.

This is strictly more conservative than the audit's "just delete `.clone()`" because it keeps
correctness on the CPU branch the audit's §4 warns about, while still removing the redundant copy
on the GPU branch (where the audit's "High" impact actually lands — multi-GB learner state dicts).

## Q2 per-site decision table

| # | Site | Source | `.clone()` decision | Why |
|---|------|--------|---------------------|-----|
| 1 | `weight_syncer/objectstore.py:70` `_to_cpu_tensor` (tensor branch) | device-agnostic; reached by CUDA learner push + CPU models | **conditional** (drop on CUDA, keep on CPU) | `.cpu()` copies iff source is CUDA; can't prove always-CUDA → device-conditional via `_independent_cpu` |
| 2 | `weight_syncer/objectstore.py:71` `_to_cpu_tensor` (non-tensor branch) | `torch.as_tensor(value)` already builds a FRESH tensor | **remove unconditionally** | `as_tensor` on a non-tensor allocates new storage; no alias to a live param exists → clone always redundant. (Still routed through `_independent_cpu` for uniformity; on the already-CPU fresh tensor it would clone, so call `.detach().cpu()` directly here — see note) |
| 3 | `weight_syncer/compression.py:44` `compress_state_dict` | input already passed through `_to_cpu_tensor` (now independent); `payload` is that tensor or its `.to(dtype)` (new tensor) | **remove unconditionally** | float branch `.to(transport_dtype)` allocates new storage; non-float branch is the already-independent `_to_cpu_tensor` output. `payload` is independent of the live param → clone redundant |
| 4 | `workers/actor/learner_worker.py:537` `_cpu_state_dict` | device-agnostic (CUDA learner / CPU tests) | **conditional** via `_independent_cpu` | same proof as site 1 |
| 5 | `workers/inference/inference_worker.py:267` `_cpu_state_dict` | device-agnostic (CUDA inference / CPU tests) | **conditional** via `_independent_cpu` | same proof as site 1 |

Note on site 2/3: the conservative win there is removing a guaranteed-redundant `.clone()` on a
tensor that is ALREADY a fresh independent allocation. Sites 1/4/5 are the device-agnostic ones
needing the conditional. `patch.py` / `bucket.py` call `_to_cpu_tensor` and inherit its fix.

Net effect on the CUDA learner/inference push path: the trailing full-model `.clone()` (one extra
host copy of every parameter) disappears, while the on-CPU path keeps its independent copy.

## Q6 design — batch the inference D2H

`inference_worker.py:115-122` currently does, per env row:
```python
"actions":       [row.detach().cpu().numpy().astype(np.float32) for row in actions],
"obs_embedding": [row.detach().cpu().numpy().astype(np.float32) for row in obs_embedding],
```
`actions` is `[N,7]` on device, `obs_embedding` is `[N,H]` on device. Each `row.cpu()` is one
D2H transfer ⇒ `2N` transfers per batch.

Replace with ONE D2H per tensor, then split on CPU:
```python
actions_np = actions.detach().cpu().numpy().astype(np.float32)        # [N,7] one D2H
obs_np     = obs_embedding.detach().cpu().numpy().astype(np.float32)  # [N,H] one D2H
return {
    "actions":       [actions_np[i] for i in range(actions_np.shape[0])],
    "obs_embedding": [obs_np[i]     for i in range(obs_np.shape[0])],
    ...
}
```
`numpy[i]` is a view into the contiguous batch array (no copy), per-env shapes/dtype unchanged.
Output is bit-identical to the per-row reference (same dtype cast, same values) — `atol=0`.

## TDD plan (RED → GREEN)

- **Q6 numeric**: build a fake worker batch (synthetic `actions`/`obs_embedding` CPU tensors),
  assert batched output `==` per-row reference list (`np.array_equal`, atol 0), shapes/dtypes
  per env preserved.
- **Q6 call-count (RED driver)**: monkeypatch `torch.Tensor.cpu` to count calls; assert the new
  code path calls `.cpu()` exactly twice (once per batch tensor), not `2N` times. RED on the
  current per-row loop, GREEN after batching.
- **Q2 independence**: for each changed helper, build a state dict of CPU tensors, run the helper,
  mutate the SOURCE in place, assert the captured tensor is UNCHANGED (proves clone-removal kept
  an independent copy on the CPU branch — the §4 case). Plus assert the non-tensor / dtype-cast
  branches also return independent tensors. No CUDA needed: the CPU branch is exactly the one that
  still clones, so it is the one whose independence must be re-proved.

All pytest/ruff via `conda run -n dreamervla`. No Ray/GPU required (helpers are import-only; the
worker D2H is exercised via a thin synthetic call, monkeypatching the modules so init isn't run).

## Out of scope
Other audit items (Q1, Q3–Q5, Q7–Q11, H*). Roadmap doc untouched.
