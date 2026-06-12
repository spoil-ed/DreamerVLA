# CLAUDE.md

This file is the Claude entrypoint for repository guidance. The canonical
agent instructions, repository orientation, extension rules, and workflow
expectations live in [AGENTS.md](AGENTS.md). For contribution mechanics, see
[CONTRIBUTING.md](CONTRIBUTING.md).

Keep this file intentionally short so Claude-specific guidance does not drift
from AGENTS.md.

## Current Routing Snapshot

- Launch the grouped Hydra entry with
  `python -m dreamer_vla.train experiment=<name> task=<suite>`.
- Shell launchers such as `scripts/train_vla.sh`, `scripts/train_wm.sh`, and
  `scripts/train_dreamervla.sh` forward ordinary `key=value` overrides to the
  same grouped entry.
- Config groups are `experiment/`, `VLA/`, `worldmodel/`, `classifier/`,
  `dreamervla/`, `evaluation/`, `task/`, and `logger/`.
- Mainline training defaults to `logger=tensorboard`; use `logger=wandb` for
  W&B online scalar metrics.
- OFT Scheme-A sidecars should match the preprocess launcher output:
  `${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h2`.

## Claude-Specific Notes

- Treat AGENTS.md as authoritative if this file and AGENTS.md ever disagree.
- Do not add new top-level route YAMLs for grouped training; use
  `experiment=<name>` plus cohesive module groups.
- Keep implementation code under `dreamer_vla/`; shell files stay thin,
  resumable launchers.
- Prefer small, tested changes that preserve the Runner pattern and existing
  Hydra composition.
