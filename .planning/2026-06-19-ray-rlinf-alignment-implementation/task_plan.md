# Ray RLinf Alignment Implementation Plan

## Goal
Implement the Ray/RLinf alignment report in testable stages, preserving DreamerVLA's optional-Ray single-machine default while closing the real backend gaps.

## Phases
1. Inventory current dirty implementation and identify gaps. Status: complete.
2. Add failing tests for real Ray learner update mode and replay classifier proxy. Status: complete.
3. Implement real cotrain learner mode and runner phase selection. Status: complete.
4. Implement and verify manual precision, FSDP, collective sync, hardware discovery, FlexiblePlacement, model registry, and resource config groups. Status: complete.
5. Run focused tests plus full unit/e2e verification. Status: complete.
6. Commit and push all alignment implementation changes. Status: in_progress.

## Decisions
- Keep Ray as an optional backend, matching DreamerVLA's mainline single-machine design.
- Keep `synthetic_ppo` for cheap Ray smoke tests.
- Reuse existing single-machine DreamerVLA update functions for real Ray learner phases.
- Treat auto-VRAM knobs as out of scope; expose RLinf-style manual resource levers instead.
