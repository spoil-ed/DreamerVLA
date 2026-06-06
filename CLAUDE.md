# CLAUDE.md

This file is the Claude entrypoint for repository guidance.

For the canonical agent instructions, repository orientation, and workflow expectations, see [AGENTS.md](AGENTS.md).

For full contribution flow, code style, and PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

Keeping the detailed guidance in `AGENTS.md` avoids duplication and prevents the two files from drifting out of sync.

**Quick orientation:** Dreamer-VLA is a single-machine multi-GPU research framework combining a VLA encoder (RynnVLA / OpenVLA-OFT / Chameleon action head) with a Dreamer-style world model (DreamerV3 RSSM, DINO-WM, TSSM) on the LIBERO benchmark. It uses **Hydra** for config and a **Runner** pattern as the training unit. Distributed runs use **torchrun** (DDP) or **FSDP**; there is no Ray, no multi-node cluster. Mainline pipeline: VLA SFT → precompute action-hidden sidecar → action-hidden WM → DreamerVLA actor-critic / WMPO outcome. Python 3.11; type hints and docstrings on public APIs. If something is unclear, add a `TODO(agent)` and note the limitation.

---

## Code structure

- **`.cursor/`** – Rules and skills: `rules/agents-md.mdc`, `skills/add-install-docker-ci-e2e`, `skills/add-example-doc-model-env`, `skills/review-pr`.
- **`.agents/`**
- **`.claude/`**
- **`.codex/`**

- **`dreamer_vla/`** – Main package (installed as `dreamer_vla`):
  - `algorithms/` – PPO, GRPO, DINO-WMPO, TD-MPC; actor, critic, reward.
  - `cli/` – Hydra entrypoints for training and evaluation.
  - `dataset/` – Offline datasets (LIBERO, OpenVLA-OFT) and online rollout dumpers.
  - `envs/` – LIBERO sim and online env wrappers.
  - `models/` – Encoder, world model, VLA backbones.
  - `runners/` – `BaseRunner` + VLA SFT, world model, classifier, DreamerVLA, eval.
  - `trainer/` – Shared DDP / FSDP helper.
  - `utils/` – Checkpoint, logger, optim, EMA, visualization, distributed helpers.
  - `preprocess/` – Dataset preprocessing and hidden extraction pipelines.

- **`configs/`** – Hydra configs, one top-level YAML per training route (VLA, WM, DreamerVLA, OFT, classifier, eval); tasks under `task/libero_*.yaml`.

- **`scripts/`** – Shell launchers and Python tools for training, evaluation, preprocessing, diagnostics, setup tools, smoke tests.

- **`tests/`** – `unit_tests/`, `e2e_tests/` (VLA, WM, DreamerVLA, OFT, classifier); e2e configs under `e2e_tests/<route>/*.yaml`.

- **`third_party/`** – Vendored upstream libraries (LIBERO, OpenVLA-OFT, robosuite, opensora, apex, etc.).

- **`data/`** – Runtime inputs, outputs, and intermediate artifacts:
  - `dataset/` – LIBERO, CALVIN, preprocessed data, task metainfo.
  - `ckpts/` – Pretrained weights.

- **`docs/`** – Repository structure, install notes, history, plans, findings, write-ups, paper draft.

- **`requirements.txt`** – Python deps; PyTorch, flash-attn, ColossalAI installed separately.

---

## How Dreamer-VLA runs

You launch one Hydra config (e.g. `python -m dreamer_vla.cli.train --config-name=dreamervla_rynn_dino_wm_wmpo_outcome task=libero_goal`). `dreamer_vla/cli/train.py` reads `RANK`/`WORLD_SIZE` from the env and forces `training.distributed_strategy=ddp` under `torchrun`, then resolves `cfg._target_` to a **Runner** class and runs `setup → execute → teardown`. The runner owns dataset, encoder, world model, actor / critic / reward, optimizer, logger, and checkpoints; there is no separate worker / scheduler layer. Online RL uses an in-process `LiberoOnlineEnv` (or the multi-proc variant in `scripts/training/train_online_rynnvla_action_hidden_dreamervla_multiproc.py`). Training backbones (DDP vs FSDP), mixed precision, gradient checkpointing, EMA, and LR schedule are config knobs under `training:`, not code branches.

---

## Configuration guides

- **Route, not knob:** pick one top-level config and override task + run-tag + GPU count; don't recombine fields by hand. Registry in [configs/README.md](configs/README.md).
- **Task switch:** `task=libero_goal | libero_object | libero_spatial | libero_10` (or any `configs/task/*.yaml`) via Hydra; task YAMLs hold dataset paths, horizons, sidecar expectations, task-specific dims.
- **One-trajectory SFT:** `CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal`; `dataset.trajectory_offset` and `dataset.demo_selection_seed` pick which demo.
- **OOM:** tune `dataloader.batch_size`, `dataloader.num_workers`, `training.gradient_accumulate_every`, `training.enable_activation_checkpointing`, FSDP mixed precision; for online RL also `env.num_envs` and chunk length. For VLA SFT, `training.vla_train_action_head_only=true` freezes the backbone.
- **Resume:** runners respect `training.resume` + `resume_dir` under `data/outputs/.../checkpoints/`; `training.resume_advance_epoch` controls the epoch counter on resume.

---

## Metrics, checkpoints, and evaluation

- **Metrics:** runners use `dreamer_vla/utils/json_logger.py` plus optional wandb / tensorboard. Online RL emits chunk-credit, KL (k1 estimator), advantage, value, action-clipping; see `dreamer_vla/algorithms/ppo/dense_chunk.py` and `outcome.py` for semantics.
- **Checkpoints:** saved under `${training.out_dir}/checkpoints/` at `training.checkpoint_every` cadence; EMA copies under `ema/` when `training.use_ema=true`.
- **Evaluation:** LIBERO rollout via `bash scripts/eval_libero_vla.sh` with `eval_libero_vla.yaml` (Dreamer ckpts: set `eval.ckpt_kind=dreamer`, `eval.dreamer_policy_source=ckpt|init`, `eval.dreamer_actor_input_source=rssm|encoder|encoder_sequence`); OpenVLA-OFT eval via `scripts/eval/launch_openvla_oft_*.sh`; closed-loop / fidelity via `scripts/diagnostics/measure_wm_*.py` and `diagnose_ppo_imagine_vs_real.py`.

---

## When things go wrong

Install (LIBERO editable install, flash-attn wheel, ColossalAI / TensorNVMe / APEX, egl_probe): see [docs/install.md](docs/install.md). Rendering: set `MUJOCO_GL=egl`; smoke-test via `scripts/smoke/smoke_libero_online_env.py`. NCCL / CUDA timeouts under DDP are usually one rank diverging (NaN, mismatched batch) — read the rank-0 log before assuming network; the DDP synchronization guards in `dreamer_vla/algorithms/ppo/outcome.py` exist for a reason, don't remove them.

---

## Key ideas and plugging in

**Config** (`configs/*.yaml`): each top-level YAML carries a `_target_` pointing at a runner class. New route = new YAML + new runner; do not add boolean switches to fork existing runners.

**Runner** (`dreamer_vla/runners/`): the training unit. Subclass BaseRunner and implement `setup` / `execute` / `teardown`; reuse its distributed-init and checkpoint plumbing instead of redoing them per runner.

**Algorithms** (`dreamer_vla/algorithms/`): PPO family, GRPO, DINO-WMPO, TD-MPC, DreamerVLA actor-critic. Each variant exposes a stable kwargs / return signature so runners compose them without conditional branching.

**Models** (`dreamer_vla/models/`): three trainable building blocks — encoder, world model, VLA backbone — each behind a protocol so a runner can swap implementations without changing the algorithm.

**Datasets** (`dreamer_vla/dataset/`): must expose a stable public API enforced by a test in `tests/`.

**Envs** (`dreamer_vla/envs/`): split between offline (dataset-driven) and online (rollout-driven) wrappers.

---

## Extending Dreamer-VLA

### New training route

Add a runner class under `dreamer_vla/runners/` subclassing BaseRunner; export it from the package init; add a YAML in `configs/` with a `_target_` pointing at the new class and `defaults: - /task: libero_<suite>`. Add a shell launcher in `scripts/` only if the invocation differs from `python -m dreamer_vla.cli.train --config-name=<my_route>`.

### New PPO variant

Add a module under `dreamer_vla/algorithms/ppo/` matching the existing kwargs / return signature; wire it into the relevant runner. Do not branch existing variants with `if algorithm == "my_variant"`. Add regression tests in `tests/` covering the invariants the variant must hold.

### New world model

Add a module under `dreamer_vla/models/world_model/` subclassing the base world model; document the forward / imagine contract; reward heads stay in the shared module; wire into a runner + YAML rather than branching existing WM modules.

### New encoder / actor

Implement the encoder protocol or subclass the base actor in their respective `dreamer_vla/models/` subdirectories; do not create a parallel hierarchy.

### New env (beyond LIBERO)

Not a stable extension surface. The data path and reward labels assume LIBERO HDF5 + task metainfo. Open an issue before starting; expect a parallel env module, parallel dataset module, per-suite metainfo, and online-RL action / observation plumbing changes.

---

## Style and contributing

Python 3.11; type hints and docstrings on public APIs. No bare `print` in training-loop code — use `dreamer_vla/utils/json_logger.py` or runner loggers. Config YAML: static only, no computed fields; derive in the runner. New behavior needs at least one test under `tests/`; keep heavy GPU runs behind `scripts/smoke/`. Commits: [Conventional Commits](https://www.conventionalcommits.org/), ~72-char imperative subject, `git commit -s` to sign off. PRs: match commit title format, fill the template, link issues; for perf-sensitive changes (PPO / WM / actor) include before/after metrics and the diagnostic script used. Expensive GPU CI is gated by the `run-ci` label. Full details: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Further reading

- [Repository structure](docs/repository_structure.md) · [Install](docs/install.md) · [Script registry](scripts/README.md) · [Config registry](configs/README.md)
- [Write-up](docs/dreamer_vla_writeup.md) · [Findings](docs/findings.md) · [History](docs/history.md) · [Task plan](docs/task_plan.md) · [TODO](docs/TODO.md)
- [Classifier revision plan](docs/classifier_revision_plan.md) · [Multicollector batched encoder plan](docs/multicollector_batched_encoder_plan.md)
- [README](README.md) · [中文 README](README.zh-CN.md) · [Git workflow tutorial](tutorial_of_git.md) · [Behavioral guidelines](CLAUDE.local.md)
