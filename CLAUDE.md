# CLAUDE.md

This file is the Claude entrypoint for repository guidance. The canonical
agent instructions, repository orientation, extension rules, and workflow
expectations live in [AGENTS.md](AGENTS.md). For contribution mechanics, see
[CONTRIBUTING.md](CONTRIBUTING.md).

Keep this file intentionally short so Claude-specific guidance does not drift
from AGENTS.md.

## Current Routing Snapshot

- Launch the grouped Hydra entry with
  `python -m dreamervla.train experiment=<name> task=<suite>`.
- Shell launchers such as `scripts/train_vla.sh`, `scripts/train_wm.sh`, and
  `scripts/train_dreamervla.sh` forward ordinary `key=value` overrides to the
  same grouped entry.
- Config groups are `experiment/`, `VLA/`, `worldmodel/`, `classifier/`,
  `dreamervla/`, `evaluation/`, `task/`, and `logger/`.
- Mainline training defaults to `logger=tensorboard_wandb`; add
  `runner.logger.wandb_mode=offline` for offline W&B, or use
  `logger=tensorboard` / `logger=wandb` for a single backend.
- OFT Scheme-A sidecars should match the preprocess launcher output, whose `history`follows the         checkpoint's `num_images_in_input` (the bundled OFT ckpts are 1-image →`${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h1`).

## RLinf Alignment Snapshot

- Learn RLinf's engineering discipline, not its process sprawl: keep the default
  path on single-machine Runner + torchrun/DDP/FSDP, and keep Ray as an optional
  backend behind explicit Hydra experiments.
- Prefer early config validation for logger backends, actor-update routes,
  sidecar paths, resume checkpoints, batch/world-size divisibility,
  horizon/chunk consistency, and token/action-hidden dimensions.
- Keep run artifacts under one root with stable places for checkpoints, logs,
  TensorBoard/W&B files, videos, diagnostics, JSONL records,
  `resolved_config.yaml`, and `run_manifest.json`.
- Treat `${training.out_dir}/checkpoints` as canonical; legacy
  `${training.out_dir}/ckpt/latest.ckpt` is resume-only compatibility.
- Use RLinf-style metric namespaces: `train/`, `eval/`, `env/`, `rollout/`,
  and `time/`.
- Add low-cost smoke/e2e configs for each mainline recipe so config, logger,
  sidecar, checkpoint, and eval behavior stay executable.

## Claude-Specific Notes

- Treat AGENTS.md as authoritative if this file and AGENTS.md ever disagree.
- Do not add new top-level route YAMLs for grouped training; use
  `experiment=<name>` plus cohesive module groups.
- Register non-Dreamer actor-update variants in
  `dreamervla/algorithms/registry.py` instead of adding new training-loop
  dispatch branches.
- Keep implementation code under `dreamervla/`; shell files stay thin,
  resumable launchers.
- Prefer small, tested changes that preserve the Runner pattern and existing
  Hydra composition.
