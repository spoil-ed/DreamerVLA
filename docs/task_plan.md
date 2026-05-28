# Task Plan: Train Action-Chunk WM

## Goal

Start training the recently modified action-chunk world model on LIBERO-goal data, using the repository's current scripts/configs and preserving logs/checkpoints.

## Phases

- [completed] Identify the correct action-chunk WM config and launch command.
- [completed] Validate required data/checkpoints and choose output tag.
- [completed] Start the training run and capture the run/session/log path.
- [completed] Check early logs for shape/config/runtime failures.
- [completed] Record command, output paths, and current status.

## Notes

- Do not touch unrelated dirty worktree changes.
- Prefer an existing training script/config over inventing a new command.
- User clarified this is the DINO-WM-based action-chunk route, not the Chameleon WM route.
- Training is running from the source-complete checkout at `/home/user01/liops/workspace/DreamerVLA` because this workspace has the current DINO-WM source files.
