# AGENTS.md

Brief for AI coding agents working on Dreamer-VLA. For full contribution flow,
code style, and PR process see [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick orientation:** Dreamer-VLA is a single-machine multi-GPU research framework combining a VLA encoder (RynnVLA / OpenVLA-OFT / Chameleon action head) with a Dreamer-style world model (DreamerV3 RSSM, DINO-WM, TSSM) on the LIBERO benchmark. It uses **Hydra** for config and a **Runner** pattern as the training unit. Distributed runs use **torchrun** (DDP) or **FSDP**; there is no Ray, no multi-node cluster. Mainline pipeline: VLA SFT → precompute action-hidden sidecar → action-hidden WM → DreamerVLA actor-critic / WMPO outcome. Python 3.11; type hints and docstrings on public APIs. If something is unclear, add a `TODO(agent)` and note the limitation.

---

## Code structure

- **`dreamer_vla/`** – Main package (installed as `dreamer_vla`):
  - `algorithms/` – PPO, GRPO, DINO-WMPO, TD-MPC; actor, critic, reward.
  - `dataset/` – Offline datasets (LIBERO, OpenVLA-OFT) and online rollout dumpers.
  - `diagnostics/` – Importable diagnostics, evaluation CLIs, smoke checks, and analysis helpers.
  - `envs/` – LIBERO sim and online env wrappers.
  - `legacy/` – Isolated non-mainline utilities for old artifacts; do not import from active configs or runners.
  - `models/` – Encoder, world model, VLA backbones.
  - `preprocess/` – Dataset preprocessing and hidden extraction pipelines.
  - `runners/` – `BaseRunner` + VLA SFT, world model, classifier, DreamerVLA, eval, DDP/FSDP helpers, and standalone online/frozen training tools.
  - `utils/` – Checkpoint, logger, optim, EMA, visualization, shared helpers.

- **`configs/`** – Hydra configs. `train.yaml` is the stable grouped
  training/eval entry. `experiment/` selects recipes and composes cohesive
  groups such as `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`,
  `evaluation/`, `task/`, and `logger/`. Keep route selection in
  `experiment=<name>` and operational logging in
  `logger=tensorboard|wandb|tensorboard_wandb`; do not revive one-off
  top-level route YAMLs for new work.

- **`scripts/`** – Resumable shell launchers only. Python implementation code lives under `dreamer_vla/` and is launched with `python -m`.

- **`tests/`** – `unit_tests/`, `e2e_tests/` (VLA, WM, DreamerVLA, OFT, classifier); e2e configs under `e2e_tests/<route>/*.yaml`.

- **`third_party/`** – Vendored upstream libraries (LIBERO, OpenVLA-OFT, robosuite, opensora, apex, etc.).

- **`data/`** – Runtime inputs, outputs, and intermediate artifacts:
  - `datasets/` – LIBERO, CALVIN, and raw benchmark assets.
  - `checkpoints/` – Pretrained weights and downloaded model assets.

- **`docs/`** – Repository structure, install notes, data layout, write-ups, paper draft.

- **`requirements.txt`** – Python deps; PyTorch, flash-attn, ColossalAI installed separately.

---

## How Dreamer-VLA runs

You launch the grouped Hydra entry (e.g. `python -m dreamer_vla.train
experiment=dreamervla_rynn_dino_wm_wmpo_outcome task=libero_goal`).
`dreamer_vla/train.py` reads `RANK`/`WORLD_SIZE` from the env and forces
`training.distributed_strategy=ddp` under `torchrun`, then resolves
`cfg._target_` to a **Runner** class and runs `setup → execute → teardown`.
The runner owns dataset, encoder, world model, actor / critic / reward,
optimizer, logger, and checkpoints; there is no separate worker / scheduler
layer. Online RL uses an in-process `DreamerVLAOnlineTrainEnv` (or the
multi-proc variant in `dreamer_vla.runners.online_dreamervla_multiproc`).
Training backbones (DDP vs FSDP), mixed precision, gradient checkpointing,
EMA, and LR schedule are config knobs under `training:`, not code branches.

---

## RLinf alignment lessons

RLinf is the reference for engineering discipline, not for process topology.
Dreamer-VLA should learn RLinf's configuration, logging, checkpointing, testing,
and documentation habits while preserving the single-machine Runner design.

- **Do not copy RLinf's Ray stack:** do not introduce Ray, Cluster,
  WorkerGroup, placement strategies, or multi-node scheduler layers into
  Dreamer-VLA. Use torchrun / DDP / FSDP plus `BaseRunner`.
- **Copy the validation mindset:** prefer an early config validation pass before
  runner setup. `dreamer_vla.config.validate_cfg` checks logger backend names,
  actor-update route names, task / experiment compatibility, sidecar path
  existence and naming, resume checkpoint shape, batch-size / world-size
  divisibility, horizon / chunk-size consistency, and token / action-hidden
  dimensions before training starts.
- **Copy the run-artifact layout:** keep each run under one root, with stable
  subdirectories for `checkpoints/`, `log/tensorboard/`, `log/wandb/`, JSONL
  logs, `video/train/`, `video/eval/`, and diagnostics. BaseRunner writes
  `resolved_config.yaml` and `run_manifest.json` at setup; avoid scattering
  runtime artifacts across unrelated data folders.
- **Copy multi-backend metric logging:** metric backends should compose as a
  list, e.g. TensorBoard plus W&B online mode. Use `BaseRunner.log_metrics` and
  normalized namespaces (`train/`, `eval/`, `env/`, `rollout/`, `time/`) rather
  than ad hoc scalar names.
- **Copy checkpoint boundaries:** runners should decide checkpoint cadence and
  directory names; components should own their model / optimizer / scheduler /
  RNG state. Prefer global-step checkpoint directories for long runs while
  keeping compatibility with existing `latest.ckpt` / top-k checkpoints.
- **Copy registry-style extension points:** algorithms, losses, reward heads,
  encoders, world models, and environment adapters should be selected through
  explicit registries, protocols, or Hydra targets. Avoid adding large
  `if algorithm == ...` branches inside training loops.
- **Copy executable config matrices:** every mainline recipe should have a
  low-cost smoke / e2e config covering the intended task, logger backend,
  sidecar style, checkpoint behavior, and eval path.
- **Copy operational documentation:** when behavior changes, update the short
  docs that operators need most: config registry, logging, resume/checkpoint,
  run-artifact layout, data cleanup, and script registry.

---

## Configuration guides

- **Route, not knob:** pick one `experiment=<name>` and override task + run-tag +
  GPU count; don't recombine fields by hand outside Hydra defaults. Registry
  in [configs/README.md](configs/README.md).
- **Task switch:** `task=libero_goal | libero_object | libero_spatial | libero_10` (or any `configs/task/*.yaml`) via Hydra; task YAMLs hold dataset paths, horizons, sidecar expectations, task-specific dims.
- **Logging switch:** grouped training defaults to `logger=tensorboard`, which
  writes local TensorBoard event files under `${training.out_dir}/log`. Use
  `logger=wandb` for W&B online mode and `logger=tensorboard_wandb` for both
  backends in parallel. Logger configs live under `configs/logger/` and route
  main-process metrics through the runner `MetricLogger`. When adding logger
  configs, follow RLinf's list-backend pattern so TensorBoard, W&B, and any
  future backend can run in parallel.
- **One-trajectory SFT:** `bash scripts/train_vla.sh
  experiment=vla_sft_one_trajectory task=libero_goal`;
  `dataset.trajectory_offset` and `dataset.demo_selection_seed` pick which
  demo.
- **OOM:** tune `dataloader.batch_size`, `dataloader.num_workers`, `training.gradient_accumulate_every`, `training.enable_activation_checkpointing`, FSDP mixed precision; for online RL also `env.num_envs` and chunk length. For VLA SFT, `training.vla_train_action_head_only=true` freezes the backbone.
- **Resume:** runners respect `training.resume` plus an explicit resume path or
  the latest checkpoint under `${training.out_dir}/checkpoints/`;
  `training.resume_advance_epoch` controls the epoch counter on resume. Older
  `${training.out_dir}/ckpt/latest.ckpt` files remain load-compatible.
- **Run artifacts:** prefer one run root per invocation. Keep logs,
  TensorBoard/W&B files, videos, diagnostics, and checkpoints under that root
  so runs can be archived, compared, resumed, or deleted without chasing
  scattered files. Canonical subdirs are `checkpoints/`, `log/tensorboard/`,
  `log/wandb/`, `video/train/`, `video/eval/`, and `diagnostics/`; canonical
  root files are `resolved_config.yaml` and `run_manifest.json`.

---

## Metrics, checkpoints, and evaluation

- **Metrics:** runners use `BaseRunner.log_metrics` / `MetricLogger` for
  TensorBoard or W&B scalar metrics, plus JSONL logs where a runner still
  persists step records. Online RL emits chunk-credit, KL (k1 estimator),
  advantage, value, action-clipping; see
  `dreamer_vla/algorithms/ppo/dense_chunk.py` and `outcome.py` for semantics.
  Prefer RLinf-style metric namespaces: `train/`, `eval/`, `env/`,
  `rollout/`, and `time/`.
- **Checkpoints:** saved under `${training.out_dir}/checkpoints/` at `training.checkpoint_every` cadence; EMA copies under `ema/` when `training.use_ema=true`. Use `BaseRunner.get_global_step_checkpoint_dir(step)` / `get_component_checkpoint_dir(component, step=...)` for RLinf-style component checkpoints instead of hand-building paths.
- **Evaluation:** LIBERO rollout via `bash scripts/eval_libero_vla.sh` with `eval_libero_vla.yaml` (Dreamer ckpts: set `eval.ckpt_kind=dreamer`, `eval.dreamer_policy_source=ckpt|init`, `eval.dreamer_actor_input_source=rssm|encoder|encoder_sequence`); OpenVLA-OFT eval via `scripts/eval/launch_openvla_oft_*.sh`; closed-loop / fidelity via `python -m dreamer_vla.diagnostics.<module>`.

---

## When things go wrong

Install (LIBERO editable install, flash-attn wheel, ColossalAI / TensorNVMe /
APEX, egl_probe): see [docs/install.md](docs/install.md). Rendering: set
`MUJOCO_GL=egl`; smoke-test via `python -m dreamer_vla.diagnostics.smoke_libero_online_env`.
NCCL / CUDA timeouts under DDP are usually one rank diverging (NaN, mismatched
batch) — read the rank-0 log before assuming network; the DDP synchronization
guards in `dreamer_vla/algorithms/ppo/outcome.py` exist for a reason, don't
remove them.

---

## Key ideas and plugging in

**Config** (`configs/`): follow the standard Hydra template pattern. The stable
training entry is `configs/train.yaml`; select recipes with
`experiment=<name>`, tasks with `task=<suite>`, and detailed knobs with ordinary
Hydra overrides. Experiments are composed from a small number of meaningful
module groups: `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`,
`evaluation/`, and `task/`. Do not split every knob into tiny groups; keep each
module YAML readable and cohesive. Compatibility top-level YAMLs may remain,
but new script and docs examples should use `python -m dreamer_vla.train
experiment=<name>`. New config-facing behavior should be validated early in a
Dreamer-VLA equivalent of RLinf's config validation layer, not discovered deep
inside a training loop.

**Runner** (`dreamer_vla/runners/`): the training unit. Subclass BaseRunner and implement `setup` / `execute` / `teardown`; reuse its distributed-init and checkpoint plumbing instead of redoing them per runner.

**Algorithms** (`dreamer_vla/algorithms/`): PPO family, GRPO, DINO-WMPO, TD-MPC, DreamerVLA actor-critic. Each variant exposes a stable kwargs / return signature so runners compose them without conditional branching. Register non-Dreamer actor-update routes in `dreamer_vla/algorithms/registry.py` and reference them from config via `algorithm.update_type`.

**Models** (`dreamer_vla/models/`): three trainable building blocks — encoder, world model, VLA backbone — each behind a protocol so a runner can swap implementations without changing the algorithm.

**Datasets** (`dreamer_vla/dataset/`): must expose a stable public API enforced by a test in `tests/`.

**Preprocessed sidecars** (`data/processed_data/`): active task configs should
consume the paths emitted by the Hydra-centered preprocess launchers. For OFT
Scheme A, `scripts/preprocess/35_oft_action_hidden.sh` writes
`${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h2`; keep task YAMLs and
docs aligned with that generated name unless an experiment explicitly
overrides the path.

**Envs** (`dreamer_vla/envs/`): split between offline (dataset-driven) and online (rollout-driven) wrappers.

---

## Extending Dreamer-VLA

### New training route

Add a runner class under `dreamer_vla/runners/` subclassing BaseRunner; export
it from the package init; add the cohesive module YAML under the matching group
(`VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`, or `evaluation/`), then add
an `experiment/<name>.yaml` recipe that overrides the relevant group. Add a
shell launcher in `scripts/` only if the invocation differs from
`python -m dreamer_vla.train experiment=<name>`.

### New PPO variant

Add a module under `dreamer_vla/algorithms/ppo/` matching the existing kwargs / return signature; register it in `dreamer_vla/algorithms/registry.py` with canonical aliases and route metadata. Do not branch existing variants with `if algorithm == "my_variant"` inside training loops. Add regression tests in `tests/` covering the invariants the variant must hold.

### New world model

Add a module under `dreamer_vla/models/world_model/` subclassing the base world model; document the forward / imagine contract; reward heads stay in the shared module; wire into a runner + YAML rather than branching existing WM modules.

### New encoder / actor

Implement the encoder protocol or subclass the base actor in their respective `dreamer_vla/models/` subdirectories; do not create a parallel hierarchy.

### New env (beyond LIBERO)

Not a stable extension surface. The data path and reward labels assume LIBERO HDF5 + task metainfo. Open an issue before starting; expect a parallel env module, parallel dataset module, per-suite metainfo, and online-RL action / observation plumbing changes.

---

## Style and contributing

Python 3.11; type hints and docstrings on public APIs. No bare `print` in training-loop code — use `dreamer_vla/utils/json_logger.py` or runner loggers. Config YAML: static only, no computed fields; derive in the runner. New behavior needs at least one test under `tests/`; keep heavy GPU runs behind `dreamer_vla/smoke/`. Commits: [Conventional Commits](https://www.conventionalcommits.org/), ~72-char imperative subject, `git commit -s` to sign off. PRs: match commit title format, fill the template, link issues; for perf-sensitive changes (PPO / WM / actor) include before/after metrics and the diagnostic script used. Expensive GPU CI is gated by the `run-ci` label. Full details: [CONTRIBUTING.md](CONTRIBUTING.md).

### Script and launcher memory

- Shell files are one-command launchers, not a code organization layer. Write
  commands in a form that can be copied into a terminal.
- Avoid shell loops, `case`, functions, and argument parsers. Use Python/Hydra
  for iteration, dispatch, resume state, GPU counting, and parameter mapping.
- Shell `if` is acceptable only for "should this step run?" checks, required
  input checks, and simple skip/resume guards that cannot reasonably live in
  Python.
- Prefer `conda activate dreamervla` followed by direct `python -m ...` or
  `uv pip install ...`; do not wrap everything in `exec "${PYTHON}" ...`.
- Install, download, preprocess, train, world-model, classifier, DreamerVLA,
  and eval entrypoints should all be Hydra-centered. Use config groups and
  CLI overrides such as `experiment=world_model_dinowm_chunk task=libero_goal
  gpus=0,1 batch_size=16 training.max_steps=1000`.

---

## Further reading

- [Repository structure](docs/repository_structure.md) · [Install](docs/install.md) · [Script registry](scripts/README.md) · [Config registry](configs/README.md)
- [Write-up](docs/dreamer_vla_writeup.md) · [Data layout](docs/data_layout.md)
- [README](README.md) · [中文 README](README.zh-CN.md)
