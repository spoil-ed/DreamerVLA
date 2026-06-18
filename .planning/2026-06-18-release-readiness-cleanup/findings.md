# Findings & Decisions

## Requirements
- User asks for a full repository review, final-release convergence, no alternate versions except TODOs, and deletion of meaningless outputs.
- Destructive cleanup is explicitly requested for meaningless outputs, but deleting tracked code/config/docs alternatives is high risk and needs a concrete policy before execution.

## Research Findings
- Current checkout is a normal `main` branch checkout, not a linked git worktree.
- Existing untracked root files `task_plan.md`, `findings.md`, `progress.md`, and `docs/experiment_tutorials/RLinf_aligned_LIBERO_rollout_execution_plan.md` belong to a prior RLinf rollout/eval plan.
- Git ignored output/runtime surfaces include `.pytest_cache/`, `.ruff_cache/`, `.venv/`, `dreamer_vla.egg-info/`, many `__pycache__/` dirs, `data/`, `wandb/`, `third_party/`, `docs/superpowers/`, local agent files, and `tutorial_of_git.md`.
- Because `data/` and `third_party/` are ignored wholesale but also mentioned as meaningful repository runtime assets in AGENTS.md, they must not be bulk-deleted without explicit user approval.
- `tests/unit_tests/test_repository_hygiene.py` already encodes prior release cleanup constraints: no archive/graveyard, no old top-level command groups, no stale route names, no local absolute paths in active files, and no generated logs tracked by git.
- `data/outputs/` is 621G and contains experiment outputs/logs/checkpoints: `vla` 563G, `dreamervla` 43G, `worldmodel` 15G, logs 1.3G, eval 3.7M.
- `data/checkpoints/` is 423G and contains downloaded/pretrained model inputs; `data/datasets/` is 729G and contains LIBERO + processed data inputs/sidecars; these should not be treated as meaningless output.
- Ignored reproducible cleanup targets observed: `.pytest_cache/`, `.ruff_cache/`, `.venv/`, `dreamer_vla.egg-info/`, `wandb/`, `logs/`, and `__pycache__/` under source/tests.
- Temporary/incomplete files observed under processed sidecars: four `*.hdf5.rank*.tmp` files in `data/datasets/processed_data/*_pi0_legacy_action_hidden_vla_policy_h2/`.
- Stale process markers observed: three `*.pid` files under `data/outputs/logs/...`.
- Ray backend release-surface offenders include `dreamervla/scheduler/`, `dreamervla/workers/`, `dreamervla/runners/*ray*`, `configs/experiment/*ray*`, `scripts/e2e_coldstart_warmup_cotrain_ray.sh`, `docs/ray_online_cotrain_backend.md`, and ray-specific tests.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use an isolated `.planning/2026-06-18-release-readiness-cleanup/` plan | Avoid mixing this cleanup with the prior RLinf plan in root planning files. |
| Treat caches/build artifacts as safe cleanup candidates | They are reproducible and ignored by git. |
| Treat model checkpoints, datasets, vendored third-party repos, and prior planning docs as protected until reviewed | They may be required assets or user work despite being ignored/untracked. |
| Remove `data/outputs/` contents but keep the directory README if present | Outputs are generated experiment artifacts and not part of final release state. |
| Do not delete docs documents | User clarified docs files should be preserved; tracked docs deletions were restored. |
| Do not prohibit Ray | User clarified Ray does not need to be banned; tracked Ray code/config/tests were restored. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Planning script failed under `sh` | Re-ran with `bash`. |
| Over-interpreted AGENTS.md Ray guidance as a hard ban and deleted tracked Ray/docs files | Restored tracked files with `git restore`; will avoid adding Ray-prohibition tests. |
| Deleted untracked `docs/experiment_tutorials/RLinf_aligned_LIBERO_rollout_execution_plan.md` | File was not tracked and no duplicate was found; root planning files still preserve the RLinf plan summary/context. |

## Resources
- AGENTS.md instructions provided in the user message.
- `README.md`, `configs/README.md`, `scripts/README.md`
- `tests/unit_tests/test_repository_hygiene.py`
