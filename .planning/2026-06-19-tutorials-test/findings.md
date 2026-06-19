# Findings & Decisions: Tutorial Tests

## Requirements
- User asked to fully test that all tutorials are okay.
- Interpret tutorials as first-party files under `docs/experiment_tutorials/`.

## Research Findings
- First-party tutorial directory: `docs/experiment_tutorials/`.
- Markdown files found:
  - `README.md`
  - `RynnVLA_LIBERO.md`
  - `OpenVLA_Onetraj_LIBERO.md`
  - `OpenVLA_Onetraj_LIBERO_action_hidden_world_model.md`
  - `OpenVLA_Onetraj_LIBERO_backbone_latent_world_model.md`
  - `OpenVLA_Onetraj_LIBERO_coldstart_rollout_collection.md`
  - `OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`
  - `RLinf_aligned_LIBERO_rollout_execution_plan.md`
- `README.md` lists the six operational tutorials; the RLinf file is an execution plan/reference, not a normal user tutorial.
- Total tutorial markdown size is 1,509 lines.
- Many commands are heavy training/preprocess/eval commands that require LIBERO datasets, OFT/Rynn checkpoints, GPU, and long runtime. The locally meaningful automated checks are static references, Hydra config composition, launcher dry-runs, and existing pytest coverage.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use pytest collection/unit tests plus targeted dry-runs | Existing tests already encode many docs/config/script contracts; dry-runs exercise tutorial launchers without starting long training. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- `docs/experiment_tutorials/`

## Visual/Browser Findings
- None.
