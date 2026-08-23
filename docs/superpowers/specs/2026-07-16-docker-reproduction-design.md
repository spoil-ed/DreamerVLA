# DreamerVLA Docker Reproduction Design

## Goal

Publish a Docker Hub image that contains the DreamerVLA source tree and its pinned
runtime environment. A user with one 8xH100 80 GB host can mount one persistent data
directory, prepare the public `libero_goal` assets, and run the complete reproduction
chain:

```text
prepare assets -> train WM for 30 epochs -> train CLS for 8 epochs
               -> select best WM/CLS checkpoints
               -> train frozen-WM/CLS Dreamer for 20,000 global steps
```

The workflow has exactly two user-facing reproduction scripts. Interrupted stages
resume automatically, and completed stages are reused only after validation.

## Scope

The first release supports one hardware and task profile:

- Ubuntu 22.04 container userspace.
- NVIDIA CUDA 12.4.1 development runtime and cuDNN.
- Eight NVIDIA H100 80 GB GPUs.
- A host NVIDIA driver compatible with the current 580.95.05 baseline.
- Python 3.11 and PyTorch 2.5.1+cu124.
- LIBERO `libero_goal` only.
- OpenVLA-OFT one-trajectory policy checkpoint from
  `Haozhan72/Openvla-oft-SFT-libero-goal-traj1`.

Supporting other GPU counts, GPU models, LIBERO suites, or training budgets is outside
the first release. Hydra overrides may remain technically possible, but the published
reproduction claim applies only to this profile.

## Image Boundary

The image is built from
`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`. It contains:

- the DreamerVLA source tree at `/opt/dreamervla`;
- a Python 3.11 environment;
- PyTorch 2.5.1+cu124, Ray 2.55.1, PEFT 0.11.0, FlashAttention
  2.7.1.post1, and the OpenVLA-OFT Transformers 4.40.1 fork;
- the pinned LIBERO, robosuite, robomimic, mimicgen, OpenVLA-OFT, and EGL probe
  source dependencies installed through the repository's existing install workflow;
- the two reproduction entrypoints and their Hydra/Python implementation;
- OCI labels for the source URL, Git commit, image version, and build time.

Because `.git` is excluded from the release image, the build also writes immutable
source metadata to `/opt/dreamervla/.dreamervla-image.json`. Runtime manifests read
the commit and image profile from this file rather than inferring them from a mutable
checkout.

The image does not contain model checkpoints, LIBERO demonstrations, generated
sidecars, or training outputs. Docker does not package the NVIDIA kernel driver; the
container receives GPUs through NVIDIA Container Toolkit.

The runtime paths are fixed:

```text
/opt/dreamervla  immutable image-owned source and environment
/data            writable host-mounted DVLA_DATA_ROOT
```

The documented launch contract uses `--gpus all`, `--ipc=host`, `--network=host`,
`--shm-size=100g`, `--ulimit memlock=-1`, and one bind mount or named volume at
`/data`. The working directory is `/opt/dreamervla`, and the default command opens a
Bash shell with the DreamerVLA environment active.

The source is intentionally included even though RLinf's current image asks users to
clone RLinf after entering the container. Including the source makes the DreamerVLA
image, dependency set, and Git commit one immutable reproduction unit. Developers may
explicitly mount a checkout over `/opt/dreamervla`, but that mode is not the published
reproduction path.

## Docker Hub Publication

The repository publishes a public Docker Hub image named
`${DOCKERHUB_NAMESPACE}/dreamervla`. `DOCKERHUB_NAMESPACE` is a required GitHub
repository variable; Docker Hub credentials are GitHub Actions secrets and never enter
the image or tracked files.

Each release publishes immutable tags for the release and Git commit, plus a moving
profile tag:

```text
${DOCKERHUB_NAMESPACE}/dreamervla:cu124-h100-v1
${DOCKERHUB_NAMESPACE}/dreamervla:<release-version>
${DOCKERHUB_NAMESPACE}/dreamervla:sha-<12-character-commit>
```

The publishing workflow builds but does not publish on pull requests. A version tag or
manual release dispatch builds, verifies, and pushes the public image. Image builds use
the public internet; runtime credentials and Hugging Face tokens are not baked into
layers.

## User-Facing Entry Points

The only new reproduction shell entrypoints are:

```text
scripts/reproduce/01_prepare_assets.sh
scripts/reproduce/02_train_dreamer.sh
```

They remain one-command launchers. They set no training defaults, contain no loops or
custom argument parser, and execute a Hydra-selected Python reproduction workflow.
All task, source, revision, budget, path, and validation values live in static Hydra
configuration. Python owns orchestration, state transitions, subprocess invocation,
checkpoint selection, and structured manifests.

The workflow calls the existing registered install, download, preprocessing, and
experiment entrypoints. It does not modify, rename, reorder, or copy logic from the
frozen files under `scripts/install/`, `scripts/download/`, or `scripts/preprocess/`.

## Asset Preparation

`01_prepare_assets.sh` performs four ordered stages.

### 1. Runtime preflight

The workflow requires:

- `/data` to be a writable mount rather than image-local ephemeral storage;
- eight visible CUDA devices, each reported as an NVIDIA H100 with at least 80 GB;
- the expected PyTorch/CUDA, Ray, Transformers fork, PEFT, and FlashAttention
  versions;
- enough free space for the checkpoint, raw demonstrations, processed HDF5 data,
  hidden-token sidecars, and subsequent training outputs;
- working Hugging Face and Git network access when an asset is absent.

Failure is immediate and reports the failed contract and corrective action. No download
starts before the preflight passes.

### 2. OpenVLA-OFT checkpoint

The workflow invokes the registered one-trajectory download entrypoint for
`Haozhan72/Openvla-oft-SFT-libero-goal-traj1`. The release pins source revision
`d20e1d447dfd87c0daa121b0739e2a379f7fe334` and verifies the downloaded file tree
against a tracked asset manifest.

Validation rejects Git LFS pointer files, empty files, missing model shards, missing
configuration/tokenizer/statistics files, the wrong action-head metadata, the wrong
dataset statistics key, and any SHA-256 mismatch. A valid existing checkpoint is not
downloaded again.

### 3. LIBERO data

The workflow invokes the registered LIBERO downloader for `libero_goal` only. The
upstream dataset repository identity is `yifengzhu-hf/LIBERO-datasets`, whose release
baseline revision is `f13aa24a3da8c43c7225569f28c562979fa0e35a`.

The tracked asset manifest contains the expected file set, sizes, and SHA-256 values.
This content fingerprint is authoritative even if the upstream default branch later
moves. Upstream drift fails explicitly instead of silently changing the experiment.
A valid existing dataset is reused without triggering the upstream helper's overwrite
prompt.

### 4. Preprocessing and validation

The workflow invokes the registered `libero_goal` preprocessing entrypoint. It creates
the reward HDF5 data and uses all eight GPUs to generate the OpenVLA-OFT hidden-token
sidecars. Existing preprocessing is reused only if the repository's structural checks
pass.

Validation proves that reward and hidden directories have the same files, demos, and
trajectory lengths; every sidecar is complete; and `preprocess_config.json` matches
the task contract. The external observation sidecar is `hidden_token [T,256,4096]`
with history 1 and storage stride (`preprocess_config.chunk_size`) 1. The downstream
task action chunk size remains 8. Decoder action slots are not accepted as an
observation sidecar.

After all stages pass, the workflow writes
`/data/reproduction/manifests/assets.json`. It records schema version, image source
commit, asset origins and revisions, file-tree hashes, preprocessing contract, and
completion time. It writes atomically only after validation succeeds.

## Training Orchestration

`02_train_dreamer.sh` first revalidates `assets.json` and the referenced artifacts. It
then executes three sequential stages. Each stage gets all eight GPUs, its own run root,
and its own ordinary DreamerVLA artifacts:

```text
/data/outputs/reproduction/libero_goal/world_model/
/data/outputs/reproduction/libero_goal/classifier/
/data/outputs/reproduction/libero_goal/dreamer/
```

### World model

The stage selects `experiment=dreamer-wm`, uses official processed `libero_goal` data,
and overrides `training.warmup_replay_epochs=30`. It keeps the configured per-rank
batch size, optimizer, BF16 precision, DDP contract, evaluation, and one-checkpoint-per-
epoch cadence. It must exit successfully after exactly 30 completed epochs.

The workflow selects the flat top-k checkpoint with the minimum configured WM loss,
validates its component payload, computes its SHA-256, and records it as the selected WM
checkpoint. `latest.ckpt` remains the resume checkpoint and is not assumed to be the
best checkpoint.

### Success classifier

The stage selects `experiment=classifier_official_upper_bound`, uses the same official
processed task data, and runs the configured 8 epochs. It retains BF16 precision,
the configured data split, validation protocol, and checkpoint cadence.

The workflow selects the flat top-k checkpoint with the maximum validation F1,
validates its classifier payload and input contract, computes its SHA-256, and records
it as the selected CLS checkpoint.

### Frozen Dreamer

The final stage selects `experiment=openvla_libero` and passes the selected WM and CLS
checkpoint paths through the existing `--wm_ckpt` and `--cls_ckpt` launcher contract.
It runs `manual_cotrain.global_steps=20000`.

Before launch, the composed config must prove all frozen-route invariants:

- `_target_` resolves to `dreamervla.runners.DreamerRunner`;
- `manual_cotrain.training_mode=failure_imagined_rl`;
- `manual_cotrain.learner_updates_enabled=false`;
- both component checkpoints exist and match the hashes selected by the warmup stages;
- the WM and CLS input/output contracts match the task sidecar metadata.

The existing Ray ActorGroup, RolloutGroup, EnvGroup, and frozen LearnerGroup
implementation remains unchanged. Dreamer trains the VLA actor while WM and CLS stay
frozen. Checkpoints, evaluation cadence, offline W&B logs, TensorBoard logs, videos,
and diagnostics retain their existing Hydra definitions.

## Resume and State

The orchestration state lives at
`/data/reproduction/manifests/training_state.json`. It is an atomic, versioned manifest,
not a second model checkpoint format. For each stage it records the resolved command,
run root, status, configured budget, last observed checkpoint, selected checkpoint,
selected metric, SHA-256, image/source commit, start time, and completion time.

On every launch:

1. A completed stage is skipped only if its run manifest, selected checkpoint, metric,
   hash, and configured budget still validate.
2. An incomplete stage with `checkpoints/latest.ckpt` resumes through the existing
   `--resume <run-root>` launcher contract.
3. A stage without a checkpoint starts once in its fixed run root.
4. A mismatched completed stage fails rather than overwriting or silently mixing runs.
5. The next stage never starts unless the preceding stage completed and its selected
   checkpoint validates.

The first release does not provide an automatic destructive restart option. A user who
intentionally wants a new run supplies a different mounted data root. This preserves
the repository rule that one invocation owns one run root and avoids accidental loss
of multi-day training.

Signals are forwarded to the active child process. The state remains `running` or
`interrupted`, and the next invocation resumes from `latest.ckpt`. A nonzero child exit
code stops the workflow immediately and is propagated to Docker.

## Logging and Provenance

W&B remains offline by default so the reproduction does not require credentials.
TensorBoard and W&B files remain below each run root. Users may sync the offline W&B
run later using the existing documented workflow.

The final training manifest connects:

- Docker image name and digest when available;
- DreamerVLA Git commit;
- fully resolved Hydra commands;
- asset manifest hash;
- selected WM and CLS checkpoint paths, metrics, and hashes;
- Dreamer final checkpoint and global step;
- hardware and software version summary.

Secrets, tokens, proxy credentials, hostnames, and absolute paths outside `/data` are
excluded from persisted manifests.

## Error Handling

Errors are grouped into actionable categories:

- **environment**: wrong GPU count/model, incompatible driver/runtime, wrong package
  fork/version, insufficient shared memory, or unwritable mount;
- **asset**: download failure, upstream drift, LFS pointer, missing file, checksum
  mismatch, or invalid model metadata;
- **preprocessing**: missing reward files, incomplete sidecars, demo/length mismatch,
  or hidden-token contract mismatch;
- **checkpoint**: missing `latest.ckpt`, malformed payload, absent top-k candidate,
  metric ambiguity, or SHA-256 mismatch;
- **resume**: run-root/config/budget mismatch or state from a different source/image
  profile;
- **training**: the existing experiment process exits nonzero or does not reach its
  configured terminal epoch/global step.

Messages identify the stage, failed path or config key, expected value, actual value,
and safe next action. Validation never deletes user data.

## Repository Changes

Implementation will add focused units with these responsibilities:

- `docker/Dockerfile`: pinned image build and runtime defaults.
- `/opt/dreamervla/.dreamervla-image.json`: image-generated source commit, version,
  profile, and build-time metadata consumed by runtime provenance checks.
- `.dockerignore`: excludes `.git`, data, outputs, caches, and local third-party trees
  from the build context.
- `.github/workflows/docker-publish.yml`: build verification and Docker Hub release.
- `scripts/reproduce/01_prepare_assets.sh`: one-command asset workflow entrypoint.
- `scripts/reproduce/02_train_dreamer.sh`: one-command training workflow entrypoint.
- `configs/scripts/reproduce/prepare_assets.yaml`: asset sources, revisions, paths,
  resource requirements, and preprocessing contract.
- `configs/scripts/reproduce/train_dreamer.yaml`: ordered experiment stages, budgets,
  selection modes, run roots, and frozen-route assertions.
- `configs/reproduction/libero_goal_assets.json`: expected public asset file-tree
  fingerprint.
- `dreamervla/launchers/reproduce.py`: Hydra composition and stage orchestration.
- `dreamervla/runtime/reproduction.py`: manifests, validation, resume decisions,
  subprocess contracts, and checkpoint selection.
- `dreamervla/diagnostics/verify_reproduction.py`: container, asset, and completed-run
  diagnostics.
- `tests/unit_tests/test_reproduction_workflow.py`: deterministic unit coverage for
  config, validation, selection, state transitions, and command construction.
- `tests/e2e_tests/test_docker_reproduction.py`: gated Docker/GPU smoke coverage.
- `docs/docker_reproduction.md`: pull, launch, storage, preparation, training, resume,
  inspection, and troubleshooting instructions.
- `README.md`, `README.zh-CN.md`, `docs/README.md`, and `scripts/README.md`: concise
  links and registry entries for the new public surface.

No cotrain worker/group implementation changes are required.

## Verification Strategy

Unit tests use temporary fake assets and checkpoints; they do not download public data
or require GPUs. They cover:

- Hydra config values for `libero_goal`, 30 WM epochs, 8 CLS epochs, and 20,000
  Dreamer steps;
- expected asset manifest parsing, path confinement to `/data`, file-set checks,
  LFS-pointer rejection, size checks, and SHA-256 mismatch reporting;
- minimum-loss WM and maximum-F1 CLS checkpoint selection with ambiguity and malformed
  filename cases;
- fresh, resume, completed-skip, interrupted, and mismatch state transitions;
- generated commands using the existing registered entrypoints and checkpoint flags;
- frozen Dreamer config assertions;
- atomic manifest writes and secret redaction.

Repository verification runs Bash syntax checks, Ruff, focused pytest, the complete unit
suite, Docker build, and container-side `scripts/install/60_verify.sh`. Pull-request CI
builds without pushing.

A gated 8xH100 smoke test validates GPU visibility, EGL, a tiny preprocessing fixture,
and dry-run commands for all three training stages. The release acceptance run performs
the real `libero_goal` asset preparation and starts each real stage on the target host.
The full 30 + 8 + 20,000 training chain is the scientific reproduction workload, not a
short CI substitute; its manifests and terminal checkpoints are the completion
evidence.

## User Experience

After installing Docker Engine, NVIDIA Container Toolkit, and the compatible NVIDIA
driver, a user needs only the public image and one persistent directory:

```bash
docker pull "${DOCKERHUB_NAMESPACE}/dreamervla:cu124-h100-v1"

docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 --volume "$PWD/dreamervla-data:/data" \
  "${DOCKERHUB_NAMESPACE}/dreamervla:cu124-h100-v1" \
  bash scripts/reproduce/01_prepare_assets.sh

docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  --ulimit memlock=-1 --volume "$PWD/dreamervla-data:/data" \
  "${DOCKERHUB_NAMESPACE}/dreamervla:cu124-h100-v1" \
  bash scripts/reproduce/02_train_dreamer.sh
```

Re-running the second command resumes the active training stage. Re-running the first
command revalidates and reuses correct assets.
