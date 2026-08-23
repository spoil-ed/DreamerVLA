# RLinf-Aligned Actor Memory Lifecycle Design

## Context

DreamerVLA colocates one FSDP Actor worker and one no-grad Rollout worker on
each GPU. The frozen imagined-RL route additionally places world-model
environment workers on the same GPU slice. The first PPO update completes, but
the next real rollout fails when `MultiStepRolloutWorker.reload_model()` tries
to move its policy back to CUDA.

The failing GPU contains an approximately 64.8 GiB Actor process plus the
Rollout process and another resident process. FSDP does not by itself coordinate
the residency of separate Ray processes. DreamerVLA currently leaves Actor
parameters, gradients, optimizer state, and allocator cache resident after PPO.
It also wraps the complete OpenVLA policy at the FSDP root, so a forward can
materialize an unnecessarily large parameter unit.

RLinf avoids this failure through three connected contracts:

1. Actor, Rollout, and Env workers explicitly load and offload models at phase
   boundaries.
2. FSDP wraps transformer and VLA submodules rather than only the root model.
3. Actor-to-Rollout synchronization consumes sharded state in bounded transfer
   units instead of exporting a full CPU snapshot during every online step.

DreamerVLA will adopt those contracts while retaining its Hydra construction,
Ray groups, frozen imagined-RL semantics, and complete checkpoint/resume
format.

## Goals

- Make consecutive cotrain steps memory-safe on colocated GPUs.
- Keep the encoder, world model, and classifier frozen in the Dreamer mainline;
  PPO remains the only optimizer update.
- Bound Actor training peak memory through layer-aware FSDP wrapping.
- Bound online policy-sync memory without weakening checkpoint or resume
  fidelity.
- Make every GPU residency transition explicit, observable, and exception-safe.
- Preserve the existing run root, global-step, RNG, optimizer, KL rollback, and
  checkpoint contracts.

## Non-Goals

- Migrating the mainline from FSDP1 to FSDP2.
- Changing PPO batch semantics, effective global batch size, or reward math.
- Training the frozen encoder, world model, or classifier.
- Editing the sibling RLinf workspace or vendored third-party code.

## Alternatives

### Rollout batch reduction only

Reducing rollout or PPO microbatch sizes can lower activation memory, but it
does not release the Actor before Rollout reload and does not fix root-only
FSDP materialization. It is not a complete solution.

### Actor offload only

Offloading the Actor after PPO fixes the immediate cross-process collision, but
the Actor still reaches an unnecessarily high training peak and online sync
still exports a complete state dict. This is a useful emergency workaround,
not the selected design.

### RLinf-aligned lifecycle and FSDP1

This is the selected design. It is the closest reference implementation, fits
the current PyTorch/checkpoint stack, and addresses residency, training peak,
and synchronization together.

### Immediate FSDP2 migration

FSDP2 can improve long-term composability, but it would simultaneously change
parameter representation, optimizer state conversion, checkpoint loading, and
weight synchronization. That expansion is unnecessary for this repair.

## Actor Residency State Machine

The Actor owns an explicit lifecycle with three valid residency states:

- `offloaded`: parameters, gradients, and optimizer tensors are on CPU.
- `parameters_loaded`: parameters required for state synchronization are on
  CUDA; gradients and optimizer remain offloaded.
- `training_loaded`: parameters, gradients, and optimizer state are available
  for PPO on CUDA.

Transitions are idempotent and guarded by the GPU's shared device lock. State
flags change only after a transfer succeeds. A failed transition leaves enough
state for a `finally` block to return to `offloaded`.

### Initialization

The Actor builds the policy, wraps it with FSDP, constructs optimizers, restores
optional checkpoint state, freezes the encoder partition, then immediately
offloads parameters, gradients, and optimizer state when
`actor.train_cfg.enable_offload` is enabled.

FSDP `cpu_offload` remains false. Phase offload is an orchestration mechanism;
FSDP CPU offload changes parameter execution semantics and is not a substitute.

### Policy synchronization

Before synchronization, the Actor offloads optimizer state if necessary and
loads parameters without loading gradients. All Actor ranks participate in the
FSDP state collectives. After the final bucket is acknowledged, the Actor
offloads parameters and gradients and clears CUDA allocator state.

### PPO training

The Actor receives and prepares trajectory data while offloaded. Only after
Rollout generation has stopped and its inference copy is offloaded does the
Actor acquire the device lock and enter `training_loaded`. PPO retains the
existing global-batch and microbatch hierarchy.

At the end of training, including zero-valid-sample and exception paths, the
Actor clears gradients with `zero_grad(set_to_none=True)`, releases temporary
batch tensors, runs Python garbage collection, and clears CUDA allocator cache.
The following policy synchronization offloads the optimizer and parameters.

### KL transaction

The frozen imagined-RL route creates its bounded-KL transaction after imagined
rollout collection, immediately before Actor training. No policy update occurs
between step-start synchronization and this point, so the transaction still
captures the exact behavior policy while avoiding an Actor reload during
Rollout residency.

Rollback continues to restore the complete policy and optimizer state. The
transaction export is a checkpoint-class operation, not the online rollout
sync path.

## Rollout and World-Model Residency

Rollout retains its inference-only offload contract:

- synchronize weights while the policy is on CPU;
- load the policy only inside generation/evaluation;
- offload in a `finally` block, including partial CUDA transfer failures;
- clear CUDA allocator state after offload.

The frozen WM environment gains the same phase contract. World-model and
classifier modules load immediately before imagined interaction and offload
after trajectory assembly. Real LIBERO environments remain CPU-rendered under
OSMesa. Before PPO starts, Rollout and WM/CLS therefore no longer occupy the
Actor's training memory budget.

The shared device lock is an admission guard, not a scheduler. Runner phase
ordering remains the primary owner of execution order, while the lock prevents
an accidental reload from overlapping another component's load.

## Layer-Aware FSDP1

The FSDP strategy gains configuration for sharding strategy, wrapping policy,
prefetch, all-gather limiting, and device placement. The mainline uses:

- `FULL_SHARD`;
- the rank-local CUDA device as `device_id`;
- mixed-precision parameter, reduction, and buffer dtypes;
- `limit_all_gathers=true`;
- conservative forward/backward prefetch defaults;
- the existing `use_orig_params` setting;
- layer-aware auto wrapping.

OpenVLA-specific module knowledge stays at the model boundary. The
`OpenVLAOFTPolicy` exposes FSDP wrap targets derived from the loaded model:

- language-model decoder block classes from `_no_split_modules`;
- VisionTransformer modules;
- Prismatic projector modules.

The generic FSDP strategy consumes this capability without branching on a
checkpoint name. Hydra may override or disable the model-provided policy.

Activation checkpointing is enabled for the Actor mainline. The policy delegates
`gradient_checkpointing_enable()` to the loaded language model and any supported
vision module. The method must fail clearly when the configured checkpointing
contract cannot be applied; silent no-ops are not accepted.

The existing default-off `no_sync()` option remains unchanged. Each microbatch
normally reduce-scatters gradients so FSDP retains only local shards.

## Bounded Actor-to-Rollout Weight Synchronization

The current patch sync performs a rank-zero full-state export, clones the full
model to CPU, retrieves the previous full snapshot, and compares every tensor
with `torch.equal`. Dense PPO changes most trainable language-model tensors, so
this work provides little benefit and creates large transient CPU and GPU
states.

The online path will follow RLinf's sender/receiver shape:

1. Runner starts the Rollout receive/apply operation before waiting for Actor
   send completion.
2. Actor obtains a sharded FSDP state dict on every rank.
3. Parameters are materialized and converted to the configured sync dtype one
   bounded bucket at a time.
4. Rank zero publishes each versioned bucket; it never retains a complete
   materialized model snapshot.
5. Rollout applies buckets directly to its CPU-resident policy and acknowledges
   the version.
6. Actor releases each bucket before materializing the next one, then offloads.

Bucket size is a static Hydra setting. A single tensor larger than the target
forms its own bucket. Version metadata is published only after every bucket is
complete, so Rollout cannot observe a partially published model.

Full state dicts remain available only for manual checkpoints, resume,
reproducibility hashes, and KL rollback.

## Checkpoint and Resume

Checkpoint helpers become lifecycle-aware:

- record the entry residency state;
- load only the state needed for the collective export;
- export rank-zero full policy and optimizer state using FSDP checkpoint APIs;
- restore the entry residency state in a `finally` block;
- preserve the current checkpoint schema and RNG payloads.

Resume restores policy and optimizer tensors before the initial offload. A
resumed Actor reaches the same `offloaded` steady state as a fresh Actor.
Checkpoint save must not leave Actor tensors resident and must not create a new
run timestamp.

## Metrics and Invariants

Each worker reports lifecycle and CUDA metrics at phase boundaries:

- loaded/offloaded state flags;
- allocated, reserved, and peak allocated bytes;
- parameter, gradient, and optimizer transfer time;
- sync bucket count, largest bucket, and total bytes;
- forced cleanup count and transition failures.

The runner asserts these invariants:

- Actor is offloaded before Rollout reload;
- Rollout and frozen WM/CLS are offloaded before Actor training load;
- online sync does not request a full state dict;
- every Actor rank enters state-dict and optimizer collectives in the same order;
- checkpoint and rollback restore their entry residency state.

## Error Handling

- Rollout and WM generation use `finally` blocks to offload after success or
  failure.
- Actor load, sync, training cleanup, checkpoint, and rollback transitions are
  exception-safe.
- A failed bucket publication does not advance the visible policy version.
- Invalid lifecycle transitions raise with role, rank, state, and requested
  transition.
- CUDA OOM errors include per-role residency and allocator metrics collected
  immediately before the failed admission.

## Test Strategy

Unit tests use small policies and mocked CUDA transfers to prove:

- initialization reaches `offloaded`;
- sync loads parameters only and returns to `offloaded`;
- PPO loads parameters plus optimizer only after rollout completion;
- normal, skipped, rollback, checkpoint, and exception paths clear gradients
  and restore residency;
- the OpenVLA capability produces decoder, vision, and projector wrap targets;
- FSDP receives `FULL_SHARD`, `device_id`, auto-wrap, mixed precision, prefetch,
  and all-gather settings;
- activation checkpointing delegates to the underlying model;
- normal online sync never calls full-state export;
- bucket publication is atomic by version and receiver-first ordering is used;
- checkpoint/resume round-trips policy, optimizer, RNG, and offload state.

Configuration tests pin the mainline offload, FSDP, activation-checkpointing,
and bucket settings. A gated two-step Ray/CUDA test exercises the actual
Actor/Rollout/WM phase sequence and records peak memory per role. The complete
verification also runs focused unit tests, cotrain contract tests, Ruff, shell
syntax checks for touched launchers, and `git diff --check`.

