# AGENTS.md

Canonical brief for AI agents working on Dreamer-VLA. Detailed reference lives in the
docs linked at the bottom; contribution mechanics live in [CONTRIBUTING.md](CONTRIBUTING.md).
This file stays short and big-picture — when in doubt, follow the invariants below and
point readers at the reference docs rather than restating detail here.

**What's done and what's left** are the companion ledgers [docs/HISTORY.md](docs/HISTORY.md)
(shipped work) and [docs/superpowers/TODO.md](docs/superpowers/TODO.md) (open work). AGENTS.md is the stable
rules/architecture; those two are the volatile state — read them together to align on any task.

## What Dreamer-VLA is

A single-machine, multi-GPU research framework that pairs a **VLA encoder**
(RynnVLA / OpenVLA-OFT / Chameleon action head) with a **Dreamer-style world model**
(DreamerV3 RSSM, DINO-WM, TSSM) on the **LIBERO** benchmark. **Hydra** drives config; a
**Runner** is the training unit. Mainline distributed runs use **torchrun (DDP)** or
**FSDP**; **Ray is an optional single-machine backend** for rollout / cotrain experiments,
not a second default topology or a multi-node layer. Python 3.11, type hints + docstrings
on public APIs.

Mainline pipeline: **VLA SFT → precompute action-hidden sidecar → action-hidden world
model (+ classifier) → DreamerVLA actor-critic / LUMOS RL**.

## Code structure

- **`dreamervla/`** — the package:
  - `algorithms/` — PPO family, GRPO, DINO-LUMOS, TD-MPC, DreamerVLA actor-critic; actor/critic/reward heads; `registry.py` for actor-update routes. LUMOS reward is selectable via `algorithm.lumos.reward_model` (registry in `algorithms/reward/`, default `sparse_outcome`); the success verifier (value source `V(e_t)=P(success)`) must satisfy `algorithms/verifier/SuccessVerifier` and is swapped via the `classifier` component's Hydra `_target_`.
  - `models/` — encoder, world model, VLA backbones (each behind a protocol).
  - `dataset/`, `preprocess/` — offline datasets + the Hydra-centered sidecar/hidden extraction pipelines.
  - `runners/` — `BaseRunner` + VLA SFT / WM / classifier / DreamerVLA / eval runners, DDP-FSDP helpers, standalone online/frozen tools, and optional Ray rollout/cotrain runners.
  - `launchers/` — multi-stage pipeline launchers (e.g. cold-start collect → warmup → cotrain) that compose `dreamervla.train` invocations.
  - `envs/` — offline (dataset-driven) and online (rollout-driven, `DreamerVLAOnlineTrainEnv`) wrappers.
  - `diagnostics/` — importable diagnostics, eval CLIs, smoke checks.
  - `scheduler/`, `workers/`, `hybrid_engines/` — optional Ray-backend primitives; keep them behind Hydra-selected runners and out of the default path.
  - `utils/` — checkpoint, logger, optim, EMA, viz, shared helpers. `legacy/` — isolated old-artifact utilities; never import from active configs/runners.
- **`configs/`** — Hydra. `train.yaml` is the grouped train/eval entry; select a recipe with `experiment=<name>` from `experiment/`, which composes the cohesive groups `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`, `evaluation/`, `task/`, and `logger/`.
- **`scripts/`** — thin, resumable shell launchers only; implementation lives under `dreamervla/` and runs via `python -m`.
- **`tests/`** — `unit_tests/` and `e2e_tests/` (the latter may spawn subprocs / real env / real ckpts).
- **`data/`** — runtime inputs/outputs (`datasets/`, `checkpoints/`, `processed_data/`, `outputs/`); **`third_party/`** — vendored upstreams (LIBERO, OpenVLA-OFT, robosuite, …).

## How it runs

`python -m dreamervla.train experiment=<name> task=<suite>` →
`train.py` reads `RANK`/`WORLD_SIZE` (forcing DDP under `torchrun`), runs an early
`dreamervla.config.validate_cfg` pass, resolves `cfg._target_` to a **Runner**, then
`setup → execute → teardown`. The runner owns dataset, encoder, world model,
actor/critic/reward, optimizer, logger, and checkpoints. DDP-vs-FSDP, mixed precision,
gradient checkpointing, EMA, and LR schedule are `training:` config knobs, not code
branches. Optional Ray runners may use the worker/scheduler primitives internally but
remain explicit `experiment=<name>` routes that share the model/data/checkpoint/metric
contracts of the non-Ray path.

## Core invariants (the part that matters most)

1. **Hydra is the source of truth.** Real training dims, model widths, horizons, batch
   sizes, sidecar names, checkpoint paths, and task behavior come from config — not from
   defaults baked into runner/worker logic. Code defaults are only for synthetic smokes,
   back-compat, or safe local fallbacks. **Asserts validate, they never decide.** Every
   parameter is set in Hydra config from the start, never chosen by an `assert`, fallback,
   or in-function constant; asserts only check that two quantities align
   (`derived == cfg.value`, `lhs == rhs`), never equal a literal.
2. **Name by role, not by artifact.** Classes/modules are named for their contract
   (`OFTBatchedDecoder`, `VecRolloutEnv`, `PixelHiddenSequenceDataset`), never for a
   concrete model/benchmark/checkpoint/sidecar — unless the class genuinely implements
   that one external boundary. Bind concrete choices through Hydra targets, registries,
   protocols, and task configs so one class serves many variants. **Keep models, datasets,
   and implementation classes decoupled** — each depends on the others only through a
   protocol / registry / Hydra target, never by importing or hardcoding a concrete sibling,
   so any model, dataset, or impl can be swapped from config alone.
   *Hydra-core construction (the rules that prevent the coupling traps we keep hitting):*
   build every component with `hydra.utils.instantiate(cfg.<component>)` — never
   `ConcreteClass(...)` in runner/algorithm/worker logic. Give constructors a
   `__init__(self, cfg=None, **kwargs)` shape so a `_target_` instantiates directly; don't
   force callers to hand-build a Config dataclass. When rebuilding from a checkpoint, route
   through ONE `_target_`-aware builder that falls back to the legacy default, so old ckpts
   load and new ones swap. Select implementations in config (compose/override), never by a
   runtime `cfg._target_ = "..."` mutation or `isinstance(x, Concrete)` branch. Keep
   "contract" params (history, dtype, sampling mode) in config too, not baked into the call.
3. **Derive downstream dims from the VLA + task, not by copying.** World-model,
   classifier, actor, replay, and sidecar dimensions flow from the selected VLA head +
   dataset/task metadata via Hydra interpolation. Don't change `wm_obs_dim` / `token_count`
   / `token_dim` / `chunk_size` / horizon from semantics alone — check the existing sidecar
   `preprocess_config.json` / HDF5 attrs first; if artifacts are absent, keep it explicit,
   validate, and mark `TODO(agent)`.
4. **Keep the two "hidden" concepts separate.** `wm_obs_dim` / `token_count` / `token_dim`
   describe the external VLA/sidecar latent the WM consumes; `model_dim` / `mlp_dim` /
   RSSM `deter/stoch/classes` / TSSM `d_model` describe internal WM width. External dims
   must match sidecar data; internal widths are architecture choices, never inferred from
   the dataset.
5. **Anything checkpoint-specific follows the checkpoint.** E.g. OFT `history` (h1/h2) is
   `num_images_in_input ÷ #cameras` — a property of how the VLA ckpt was SFT'd — derived
   from the single source `task.openvla_oft.expected_history`. Do not hardcode such values
   as a fixed "scheme" anywhere.
6. **Optional components are opt-in.** Build/validate what the active config declares; a
   route that doesn't define a reward worker, classifier, sidecar field, or logger backend
   simply doesn't use it. No defensive "not supported because X is missing" branches —
   prefer registries, protocol capability checks, and narrow validation of declared fields.
7. **One run, one root.** Each invocation writes under `${training.out_dir}`:
   `checkpoints/`, `log/tensorboard/`, `log/wandb/`, `video/{train,eval}/`, `diagnostics/`,
   plus `resolved_config.yaml` and `run_manifest.json` (written by BaseRunner). Don't
   scatter artifacts across unrelated folders.

## Running & extending

- **Recipe, not knobs:** choose one `experiment=<name>`, switch tasks with
  `task=libero_goal|libero_object|libero_spatial|libero_10`, and override run-tag / GPU
  count / batch size as ordinary Hydra keys. Registries: [configs/README.md](configs/README.md),
  [scripts/README.md](scripts/README.md).
- **Logging:** defaults to `logger=tensorboard_wandb` (W&B online; `runner.logger.wandb_mode=offline`
  for local). Route metrics through `BaseRunner.log_metrics` with namespaces
  `train/ eval/ env/ rollout/ time/`; no ad-hoc scalar names, no bare `print` in
  training-loop code (use the runner logger / `utils/json_logger.py`).
- **Checkpoints / resume:** saved under `${training.out_dir}/checkpoints/` at
  `training.checkpoint_every`; resume via `training.resume` + an explicit path or the
  latest there. Use `BaseRunner.get_global_step_checkpoint_dir` / component-checkpoint
  helpers rather than hand-built paths.
- **Eval:** LIBERO Dreamer rollout via `scripts/eval_libero_vla.sh`
  (`eval.ckpt_kind=dreamer`, `eval.dreamer_policy_source`, `eval.dreamer_actor_input_source`);
  raw OpenVLA-OFT via `scripts/eval/launch_openvla_oft_*.sh`.
- **New route:** subclass `BaseRunner` (implement `setup`/`execute`/`teardown`, reuse its
  distributed + checkpoint plumbing), export it, add the cohesive group YAML + an
  `experiment/<name>.yaml`; add a shell launcher only if the call differs from
  `python -m dreamervla.train experiment=<name>`.
- **New algorithm / WM / encoder:** add a module matching the existing kwargs/return (or
  protocol), register actor-update routes in `algorithms/registry.py` (referenced via
  `algorithm.update_type`), and wire through config — never `if algorithm == ...` branches
  in training loops. Add regression tests under `tests/`.
- **New env beyond LIBERO:** not a stable surface (data path + reward labels assume LIBERO
  HDF5 + task metainfo); open an issue first.
- Operational detail — OOM knobs, exact resume flags, parameter meanings — lives in
  [docs/PARAMETERS.md](docs/PARAMETERS.md) and the [tutorials](docs/experiment_tutorials/);
  don't duplicate it here.

## RLinf discipline

RLinf is the reference for *engineering discipline*, not process sprawl: early config
validation, one-root run artifacts, list-composed metric backends, registry-style
extension points, and a low-cost smoke/e2e config per mainline recipe. Adopt those habits
while keeping the single-machine Runner design and Ray as an optional, contract-sharing
backend.

## When things go wrong

- Install (LIBERO editable, flash-attn, ColossalAI/TensorNVMe/APEX, egl_probe):
  [docs/install.md](docs/install.md). OpenVLA-OFT needs the moojink transformers fork —
  installed into the main env and FATAL-checked by `scripts/install/60_verify.sh`.
- Rendering: `MUJOCO_GL=egl`, or `MUJOCO_GL=osmesa` (+ `PYOPENGL_PLATFORM=osmesa`) where
  EGL crashes robosuite `read_pixels`. Smoke: `python -m dreamervla.diagnostics.smoke_libero_online_env`.
- NCCL/CUDA timeouts under DDP usually mean one rank diverged (NaN, mismatched batch) —
  read the rank-0 log before blaming the network, and don't remove the DDP sync guards in
  `dreamervla/algorithms/ppo/outcome.py`.

## Style

Python 3.11; type hints + docstrings on public APIs. Static config YAML (no computed
fields — derive in the runner). New behavior needs a test under `tests/`; heavy GPU runs
go in `tests/e2e_tests/` (gated). Shell files are one-command launchers — no loops,
`case`, functions, or arg parsers (use Python/Hydra for iteration, dispatch, resume, GPU
counting); `if` only for run/skip/required-input guards. Commits: Conventional Commits,
~72-char imperative subject, `git commit -s`. Full mechanics: [CONTRIBUTING.md](CONTRIBUTING.md).

## Further reading

[**History (done)**](docs/HISTORY.md) · [**TODO (open)**](docs/superpowers/TODO.md) ·
[Repository structure](docs/repository_structure.md) ·
[Install](docs/install.md) ·
[Config registry](configs/README.md) ·
[Script registry](scripts/README.md) ·
[Parameter reference](docs/PARAMETERS.md) ·
[Experiment tutorials](docs/experiment_tutorials/) + [explained](docs/experiment_tutorials/EXPLAINED.md) ·
[Data layout](docs/data_layout.md) ·
[README](README.md) / [中文](README.zh-CN.md)
