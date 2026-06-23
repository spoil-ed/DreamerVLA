# PERF-Q1 — fuse the EMA update with `torch._foreach_*` (bound the per-step Python loop)

## Problem (audit Q1 / §H / §3.7, `utils/ema.py:41-45`)
`EMAHelper.step` updates the EMA shadows with a Python `for`-loop that issues two
separate CUDA kernels **per parameter**:
```python
for name, param in model.named_parameters():
    shadow = self.shadow.get(name)
    if shadow is None:
        continue
    shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
```
For a multi-billion-parameter VLA this is hundreds-to-thousands of tiny
`mul_`/`add_` launches every optimizer step — pure launch overhead on the hot path.
The fix flagged in the audit is to fuse the elementwise math into multi-tensor
(`torch._foreach_*`) kernels.

## Exact change
Replace the per-parameter loop in `EMAHelper.step` (the `else` branch after the
`update_after_step` early return) with the fused two-call form:
```python
torch._foreach_mul_(dst, decay)                 # dst *= decay
torch._foreach_add_(dst, src, alpha=1 - decay)  # dst += (1 - decay) * src
```
applied over the list of shadow tensors (`dst`) and the matching detached live
params (`src`).

This is the **exact same** elementwise math as the current loop
(`shadow = shadow*decay + (1-decay)*param`), per tensor, in the same op order
(`mul_` then `add_` with the same `alpha`). It is a launch-fusion only — numerically
identical, not an approximation.

### Constraint: `_foreach_` requires same dtype+device per call
`torch._foreach_*` kernels operate on a homogeneous list. A model can carry mixed
dtypes (fp32 master + bf16/fp16 buffers) and, under FSDP, rank-local shards live on
one device but nothing forbids mixed dtype. So we **bucket** the (shadow, param)
pairs by `(dtype, device)` and issue one fused `mul_`/`add_` pair per bucket. Within a
bucket every tensor receives the identical two ops it received in the old loop →
bit-identical result.

Iteration order is preserved by building buckets in `model.named_parameters()` order
(the same order the old loop used); we only group, never reorder the math applied to
any individual tensor.

The `update_after_step` warmup branch (the `copy_` path) is unchanged — it is a copy,
not an EMA blend, and is not on the audit's hot-path target.

## TDD steps
1. **RED** — `tests/unit_tests/test_ema_foreach_update.py`:
   - Build a small `nn.Module` with several parameters of **mixed shapes and dtypes**
     (fp32 + fp64) on CPU so the bucketing path is exercised.
   - Two independent `EMAHelper`s seeded from clones of the same params.
   - Reference: run N `step()`s where the EMA blend is computed by an explicit
     per-parameter `shadow.mul_(decay).add_(param, alpha=1-decay)` reference loop.
   - Subject: run N `step()`s through `EMAHelper.step` (the implementation under test).
   - Drive `update_after_step=0` so the fused blend branch runs every step; mutate the
     model params between steps so successive updates differ.
   - Assert every shadow tensor is **exactly equal** (`torch.equal`, atol=0): same two
     ops, same order, same dtype per tensor ⇒ bit-identical. (If a future change
     reorders ops, switch to a tight `atol` with a comment — not needed here.)
   - Also assert the warmup `copy_` branch still matches the live params when
     `optimization_step <= update_after_step`.
   - Run RED **before** editing `ema.py`: the test encodes the contract; write the
     reference loop to mirror today's math so RED→GREEN proves equivalence of the fused
     form, and guards against a dtype-bucketing regression.
2. **GREEN** — implement the `_foreach_` bucketing in `EMAHelper.step`; rerun → pass.
3. `conda run -n dreamervla python -m pytest tests/unit_tests/test_ema_foreach_update.py -q`
4. `conda run -n dreamervla ruff check dreamervla/utils/ema.py tests/unit_tests/test_ema_foreach_update.py`

## Equivalence gate
- atol = 0 (`torch.equal`) for every shadow tensor after N updates, across mixed
  dtypes/shapes. The change MUST NOT alter the EMA values — it only fuses kernel
  launches. Any nonzero diff fails the gate.

## Scope
`dreamervla/utils/ema.py` + `tests/unit_tests/test_ema_foreach_update.py` + this plan
doc only. No roadmap/other-plan edits, no unrelated code.

## Out of scope
Vectorizing the warmup `copy_` branch, FSDP-specific multi-device test coverage
(CPU mixed-dtype bucketing already exercises the grouping logic), and any other audit
item.
