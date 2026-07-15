# FSDP Gradient-Sync Memory Design

## Problem

The imagined-rollout PPO actor uses FSDP1 and accumulates 256 microbatches per
optimizer step. It currently enters `FSDP.no_sync()` for every non-final
microbatch. FSDP therefore retains full, unsharded gradients during accumulation.
For the dense OpenVLA language model this creates an approximately 14 GiB
per-rank allocation and causes the first PPO backward pass to run out of memory.

## Design

Gradient synchronization during accumulation is opt-in and disabled by default,
matching RLinf. With the default, every microbatch backward performs FSDP
reduce-scatter and accumulates only the local gradient shard. The existing loss
division, one `zero_grad()` per global batch, and one optimizer step per global
batch remain unchanged, so the effective PPO update is unchanged apart from the
order of distributed floating-point reductions.

Add `enable_gradient_accumulation: false` to the FSDP configuration contract.
The Actor consults the prepared FSDP strategy before choosing `no_sync()`.
Explicitly setting the option to `true` retains the current communication-saving,
high-memory behavior for smaller models or larger-memory systems.

All-masked microbatches continue to run the zero-valued forward/backward path.
Skipping backward on only some ranks is unsafe because FSDP collectives must be
entered consistently by every rank.

## Alternatives Considered

- Reducing `micro_batch_size` lowers activations but does not remove the full
  unsharded gradient allocation, so it is not the primary fix.
- Enabling CPU offload lowers GPU memory at substantial throughput cost and is a
  fallback rather than the default.
- Removing gradient accumulation or shrinking the effective global batch changes
  PPO optimizer semantics and is out of scope.

## Tests

Unit tests cover both branches: synchronization is used for every microbatch by
default, while an explicit opt-in uses `no_sync()` only for non-final
microbatches. Existing Actor PPO tests verify loss scaling and optimizer-step
counts remain intact.
