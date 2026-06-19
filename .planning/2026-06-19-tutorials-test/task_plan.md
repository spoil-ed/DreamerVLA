# Task Plan: Test Experiment Tutorials

## Goal
Verify that all first-party tutorial documents under `docs/experiment_tutorials/` are structurally correct and that their runnable commands/config references work in the current repository as far as local assets allow.

## Current Phase
Phase 4

## Phases

### Phase 1: Tutorial Inventory
- [x] List all tutorial files.
- [x] Extract command/config references.
- [x] Document scope in findings.md.
- **Status:** complete

### Phase 2: Static Validation
- [x] Check referenced scripts, configs, modules, and docs paths exist.
- [x] Check Markdown links where practical.
- [x] Run existing repository tests that cover tutorial/docs hygiene.
- **Status:** complete

### Phase 3: Command Dry-Runs And Lightweight Execution
- [x] Run tutorial dry-run commands where available.
- [x] Compose Hydra configs referenced by tutorials.
- [x] Avoid heavy GPU/LIBERO training unless explicitly lightweight or gated.
- **Status:** complete

### Phase 4: Full Tutorial Verdict
- [x] Summarize pass/fail/skip status per tutorial.
- [x] Record any environment-gated gaps.
- [x] Give the user concrete commands and outcomes.
- **Status:** complete

## Key Questions
1. Which files count as first-party tutorials?
2. Which tutorial commands are safe and meaningful to run locally?
3. Which checks are blocked by real checkpoint, dataset, GPU, or long training requirements?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Scope first-party tutorials to `docs/experiment_tutorials/*.md` | Other matches are vendored third-party notebooks/examples and not DreamerVLA tutorials. |
| Prefer dry-run/config/pytest validation before heavy training | The request is to validate tutorials; long GPU training/eval should be reported separately if assets or time gate it. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `scripts/train_wm.sh ... task.openvla_oft.action_hidden_dir=... dry_run=true` failed with `Key 'openvla_oft' is not in struct` | 1 | Root cause: launcher parser treated dotted `task.*` overrides as launcher `task`; added failing test and changed parser to only capture exact launcher keys plus `env.*`. |
