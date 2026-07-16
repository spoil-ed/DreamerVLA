# Online KIR Implementation Plan

## Goal

Implement WoVR-style keyframe-initialized imagined rollouts from DreamerVLA's
collected and current-step failed real trajectories, without creating a static
`.npy` initialization pool.

## Phases

1. **Design and contract audit** — in progress
   - Freeze the replay-to-WM initialization contract.
   - Verify chunk-WM history/action/proprio alignment and Hydra ownership.
2. **Replay sampler TDD** — pending
   - Add failing tests for failed-endpoint selection, histories, fallback, and grouping.
   - Implement the minimum replay sampling behavior.
3. **WM environment initialization TDD** — pending
   - Add failing tests proving real histories replace repeated-latent/zero-action bootstrap.
   - Extend the initialization contract through ReplayWorker and EnvWorker.
4. **Hydra and validation TDD** — pending
   - Add explicit KIR mixture configuration and composition/validation tests.
5. **Verification and handoff** — pending
   - Run focused unit tests and relevant regression suites in `dreamervla`.
   - Inspect diff, update documentation, commit with sign-off, and report commands.

## Decisions

- KIR candidates are failed real episodes only.
- The keyframe is the final valid real transition (`finish_step - 1`).
- Initialization carries the complete WM-required latent, action, and proprio history.
- Ordinary episode-start initialization remains available as a configured mixture.
- Cold-start collected data and current-step online data use the same replay contract.
- No `.npy` files or parallel KIR storage path will be introduced.

## Errors

| Error | Attempts | Resolution |
|---|---:|---|
