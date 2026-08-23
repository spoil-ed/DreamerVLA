# Repository Release Polish Design

## Context

DreamerVLA now has a published Docker image, a native installation path, two
reproduction entrypoints, and English and Chinese root READMEs. The execution
surface is usable, but the public documentation still mixes three different
claims:

1. what the release workflow is configured to run;
2. what has been directly validated during release preparation; and
3. what would count as a completed scientific reproduction.

The distinction matters because environment installation, Docker publication,
single-GPU startup, and checkpoint resume can be verified without claiming that
the full 30-epoch WM warmup and 20,000-step Dreamer job has finished or reproduced
reported metrics. The detailed Docker guide is also Chinese-only even though the
default repository language is English, and the public commands use a convenient
mutable image tag without showing the immutable release identity.

## Goal

Make the repository landing page and reproduction documentation concise,
bilingual, internally consistent, and explicit about release evidence. A new user
should be able to choose Docker or native setup, verify exactly which source and
image they received, prepare external assets, start or resume training, and
understand which claims have and have not been validated.

## Non-goals

- Do not change model code, training behavior, Hydra defaults, checkpoint formats,
  download behavior, or the frozen bootstrap scripts.
- Do not claim final scientific metrics or full-training completion without a
  corresponding retained artifact.
- Do not copy model weights or datasets into the Docker image.
- Do not expand this pass into a general architecture rewrite or a marketing-site
  redesign.

## Approaches Considered

### 1. Release-correctness pass (selected)

Keep the existing short reproduction-first README structure, but add a compact
project orientation, an evidence/status section, immutable release identifiers,
and consistent English/Chinese detailed guides. Extend repository tests to enforce
the public contract.

This approach directly addresses reproducibility and trust while keeping the
change reviewable and avoiding unrelated training changes.

### 2. Full landing-page redesign

Add badges, diagrams, benchmark tables, model descriptions, citations, and a broad
documentation portal. This could make the project look more complete, but it would
make the first-run path longer and would require scientific results and citation
details that are outside the current release evidence.

### 3. Implementation cleanup alongside documentation

Refactor launchers, configs, and workflows while rewriting the docs. This creates
unnecessary regression risk because the current public entrypoints already work
and the user-facing problem is primarily one of clarity and evidence.

## Public Information Architecture

The English `README.md` remains the default landing page and
`README.zh-CN.md` remains its content-equivalent Chinese companion. Both use the
same section order:

1. project purpose and the four-stage release workflow;
2. release and validation status;
3. requirements;
4. Docker reproduction;
5. native reproduction;
6. resume and outputs;
7. image contents and external assets;
8. direct entrypoints, logging, and detailed documentation.

The landing pages stay task-oriented. Architecture details continue to live under
`spec/`, command details under `scripts/README.md`, and runtime paths under
`docs/data_layout.md`.

The current Chinese `docs/docker_reproduction.md` becomes an English default
detailed guide. Its Chinese content moves to
`docs/docker_reproduction.zh-CN.md`. Each document links to its companion, and the
documentation index links to both.

## Truthful Validation Language

The docs use an explicit evidence matrix rather than the ambiguous phrase “fully
reproduced”:

- **Published and checked:** the Git commit and three Docker tags exist; all three
  image tags resolve to the same digest; OCI revision and embedded image metadata
  match the source commit; the image contains the source and required third-party
  trees; dependency imports and the unit suite pass.
- **Startup/resume checked:** only claim the single-GPU launch and checkpoint-resume
  paths when their existing retained evidence or tests can be identified during
  implementation review.
- **Configured workflow:** WM runs for 30 epochs, CLS for 8 epochs, and frozen-WM/CLS
  Dreamer for 20,000 global steps.
- **Not claimed:** completion of the entire release-sized job or reproduction of
  final scientific metrics.

If startup/resume evidence cannot be located, the docs describe those paths as
supported behavior rather than completed validation. No wording may promote a
configured target into a completed experimental result.

## Immutable Release Identity

The friendly command continues to use:

```text
spoil/dreamervla:cu124-h100-v1
```

For exact reproduction, the docs also publish:

```text
Git commit: c129946056a20df111a69e465f5966803e67abc5
Docker tag: spoil/dreamervla:sha-c129946056a2
Docker digest: sha256:6dab825adb6a9730d90e95e90821f2391d034d8a1c7285302ca5393a95c7a5c4
```

The README includes short commands to inspect the pulled image revision. It treats
the digest as the immutable identity, the SHA tag as a version-specific convenience,
and `v1` and `cu124-h100-v1` as release aliases. Native instructions show how to
check out the matching source commit for strict release reproduction without
forcing ordinary contributors into detached HEAD state.

## Docker and Asset Boundary

All public docs consistently state:

- the image contains the DreamerVLA source, CUDA/Python dependencies, and pinned
  third-party repositories;
- OpenVLA weights, LIBERO datasets, preprocessed sidecars, checkpoints, logs, and
  outputs remain under the mounted `/data` directory;
- deleting a container does not delete mounted data;
- users must not mount over `/opt/dreamervla` or provide a host `third_party` tree;
- a full release-sized job targets 8xH100 80 GB, while any separately documented
  smoke/startup check is not a performance or full-run guarantee.

## Native Installation Consistency

The root README and `docs/install.md` use the same shell initialization sequence:

```bash
bash scripts/install_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dreamervla
bash scripts/install/60_verify.sh
```

The native path uses the same asset and training entrypoints as Docker. Detailed
docs do not introduce alternative defaults or duplicate training logic.

## Contract Tests

Focused tests will read the public Markdown and release workflow files and assert:

- English/Chinese companion links exist and point to real files;
- both landing pages contain the same WM, CLS, and Dreamer schedule;
- the friendly image tag, immutable SHA tag, Git commit, and digest remain aligned;
- validation wording distinguishes configured workflow from completed results;
- both Docker guides mention the external `/data` boundary, source/third-party
  inclusion, resume behavior, and public reproduction scripts;
- native installation instructions include Conda shell initialization and the
  install verifier;
- all relative Markdown links in the modified documentation resolve.

Tests should validate stable public contracts, not punctuation or complete prose,
so future editorial changes remain inexpensive.

## Verification

Implementation verification consists of:

1. focused documentation and reproduction workflow unit tests;
2. the full unit suite;
3. `git diff --check` and a relative-link scan;
4. remote inspection of all three Docker tags and the OCI revision;
5. a clean Git worktree after committing and pushing.

No Docker rebuild is required because documentation and tests do not change the
published runtime. A later source-image release can pick up the polished docs while
reusing the existing dependency layer.

## Acceptance Criteria

- A first-time reader can identify the recommended and exact-reproduction commands
  without reading internal specifications.
- English is the default at every public documentation level, with direct Chinese
  switching links.
- Every statement about validation is backed by an observed command, test, or
  retained artifact.
- The public 30/8/20,000 schedule and frozen WM/CLS boundary remain unchanged.
- Docker and native commands call the same registered reproduction scripts.
- All focused and full unit tests pass, and no training implementation is modified.
