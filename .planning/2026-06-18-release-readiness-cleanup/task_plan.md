# Task Plan: Release Readiness Cleanup

## Goal
Audit DreamerVLA for release readiness, converge obvious non-final alternatives into documented TODOs, and remove meaningless generated outputs without deleting useful data or user work.

## Current Phase
Phase 1

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints
- [x] Document in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define deletion and consolidation policy
- [x] Treat user's "继续" as approval to proceed with safe destructive cleanup
- **Status:** complete

### Phase 3: Implementation
- [ ] Remove approved meaningless outputs
- [ ] Consolidate approved duplicate/alternative release surfaces
- [ ] Convert unavoidable alternatives to TODO notes
- **Status:** in_progress

### Phase 4: Testing & Verification
- [ ] Verify requirements met
- [ ] Document test results
- **Status:** pending

### Phase 5: Delivery
- [ ] Review outputs
- [ ] Deliver to user
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Protect existing root planning files | They are untracked but contain a separate RLinf rollout plan and should not be overwritten by this cleanup. |
| Delete only runtime outputs/caches by default | User asked to remove meaningless output; checkpoints, datasets, and vendored third-party code are inputs/assets, not disposable output. |
| Collapse Ray collector/backend from release surface | AGENTS.md says DreamerVLA should not introduce Ray/Cluster/WorkerGroup/scheduler layers; release should keep torchrun/DDP/FSDP plus Runner. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| `sh init-session.sh` hit `[[: not found]` and reused root planning files | Re-ran the same skill script with `bash`, creating isolated `.planning/2026-06-18-release-readiness-cleanup/`. |
