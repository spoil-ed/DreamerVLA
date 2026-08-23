# DreamerVLA-2 Native Environment Reproduction Design

## Goal

Reproduce the repository's native Conda setup in a new environment named
`dreamervla-2`, execute every setup stage rather than trusting existing markers,
and prove that the published LIBERO goal pipeline can enter real WM, classifier,
and Dreamer training. The work must leave a detailed, repository-visible record
of the exact machine, versions, commands, evidence, failures, and fixes.

Completing the published 30/8/20,000 training budgets is outside this validation.
Each training stage must nevertheless execute real forward/backward/update work,
emit metrics, and write a checkpoint that the next stage can consume.

## Approach

Use the repository's native staged installer, not a hand-built Conda clone and not
the Docker image. Select `dreamervla-2` through the existing Hydra-owned
`env.CONDA_ENV_NAME` setting and run every `00` through `60` stage with `force=true`.
This exercises the source of truth while keeping the pre-existing `dreamervla`
environment available as a reference.

The repository already contains several pinned third-party checkouts. The actual
installer will still fetch, check out, and install those sources. Separately,
clone each required repository into a `mktemp` directory and check out the pinned
revision to demonstrate that a clean network/source acquisition works. Temporary
clones are audit evidence only; editable installations must continue to point at
the persistent checkout under `third_party/`.

## Execution and Evidence

The implementation has five evidence layers:

1. Host baseline: OS, kernel, driver, eight H100 devices, disk, Conda version, Git
   revision, and pre-run environment inventory.
2. Setup transcript: one log per forced installer stage, plus temporary clean-clone
   revision output for all mandatory third parties.
3. Environment snapshot: Python/Torch/CUDA versions, critical distribution
   versions, `conda list --explicit`, `pip freeze --all`, `pip check`, import
   locations, OpenVLA-OFT fork signature, and FlashAttention import/CUDA kernel
   probe.
4. Workflow evidence: public asset preparation/validation followed by bounded WM,
   classifier, and Dreamer runs. Bounds are expressed as Hydra overrides; dry-run
   output does not count as evidence.
5. Repository quality gates: focused setup/reproduction tests, all unit tests,
   Ruff, and applicable Ray/LIBERO/cotrain smoke tests.

Raw transcripts and machine snapshots live below
`data/reproduction/environment/dreamervla-2/` because they are host-specific and
can be large. A concise durable record lives in `docs/` and contains the commands,
key output, artifact paths and hashes, observed dependency gaps, and exact fixes.

## Bounded Training Contract

The public `scripts/reproduce/02_train_dreamer.sh` remains the workflow owner.
For environment validation, its stage budget keys are overridden without changing
the release defaults:

- WM uses `training.warmup_replay_max_steps=1` and must write a loss-selected
  checkpoint.
- Classifier uses `training.max_train_steps=1` and must write an F1-selected
  checkpoint.
- Dreamer uses `manual_cotrain.global_steps=1` and must load both selected
  checkpoints, perform the real Ray/FSDP route, and write `latest.ckpt`.

If a runner cannot produce its normal selection checkpoint under that bound, use
the smallest runner-supported bound that does so and record the reason. Do not
replace real models, real data, Ray workers, or LIBERO interaction with fixtures.

## Dependency Repair Policy

Do not pre-emptively add packages from an existing developer environment. A setup
change is justified only by a reproducible failure in `dreamervla-2`, followed by
root-cause tracing to the owning install stage. Add the narrowest compatible pin
to `requirements.txt`, `pyproject.toml`, or the relevant `scripts/install/` stage,
then add a failing contract test before the fix and rerun the failed stage from a
clean-enough state.

Transient network, disk, GPU occupancy, or external-service failures are recorded
as operational failures and are not disguised as Python dependency changes.

## Documentation Changes

Add a native environment reproduction report under `docs/`, link it from the docs
index, and update `docs/install.md` so a custom environment name is reproducible:

```bash
bash scripts/install_env.sh env.CONDA_ENV_NAME=dreamervla-2
CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh
conda activate dreamervla-2
```

The report distinguishes confirmed passes from commands that were blocked, and it
does not claim completion based only on package installation or configuration
composition.

## Safety and Cleanup

Preserve all unrelated workspace state and never edit `third_party/`. Clean-clone
temporary directories may be removed after their revisions are recorded. Keep
training outputs and logs under the configured `data/` tree. Do not delete or
replace existing datasets, checkpoints, Conda environments, or reproduction
outputs. If an existing target is incomplete, diagnose and choose a new isolated
output path instead of overwriting it.

## Acceptance Criteria

- `conda env list` shows `dreamervla-2` with Python 3.11.
- Every installer stage has fresh exit-status evidence; `60_verify.sh` passes in
  `dreamervla-2`.
- Clean temporary clones resolve to every configured mandatory third-party commit.
- `pip check`, critical imports, CUDA visibility, the custom Transformers fork,
  PEFT compatibility, and FlashAttention verification pass.
- Asset validation/preprocessing completes and writes a complete manifest.
- WM, classifier, and Dreamer each perform real bounded training and write the
  expected checkpoint consumed by the next stage.
- Focused tests, full unit tests, Ruff, and relevant environment smokes have fresh
  results.
- Confirmed missing dependencies are fixed in setup with regression coverage.
- The durable reproduction report records exact commands and evidence, including
  any remaining limitation rather than concealing it.
