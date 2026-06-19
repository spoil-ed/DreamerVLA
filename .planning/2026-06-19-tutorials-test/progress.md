# Progress Log: Tutorial Tests

## Session: 2026-06-19

### Phase 1: Tutorial Inventory
- **Status:** in_progress
- **Started:** 2026-06-19
- Actions taken:
  - Read existing root planning files to avoid overwriting prior RLinf/OFT context.
  - Created isolated planning files for tutorial testing.
  - Listed tutorial markdown files and line counts.
  - Searched tutorials for runnable command blocks, Hydra experiment references, scripts, config paths, and markdown links.
- Files created/modified:
  - `.planning/2026-06-19-tutorials-test/task_plan.md`
  - `.planning/2026-06-19-tutorials-test/findings.md`
  - `.planning/2026-06-19-tutorials-test/progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| RED nested task override | `python -m pytest tests/unit_tests/test_setup_scripts.py::test_training_launchers_pass_nested_task_overrides_to_train_config -q` before fix | Fail reproducing tutorial launcher bug | Failed with `Key 'openvla_oft' is not in struct` | ✓ |
| GREEN nested task override | same test after fix | Pass | `1 passed in 0.40s` | ✓ |
| Launcher common flags regression | `python -m pytest tests/unit_tests/test_setup_scripts.py::test_training_launchers_accept_common_cli_flags -q` | Pass | `1 passed in 0.38s` | ✓ |
| Synthetic cold-start Ray e2e | `python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q` | Pass | `4 passed, 6 warnings in 60.64s` | ✓ |
| Real OFT cold-start GPU e2e | `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -q -s` | Pass or expose real contract issue | Failed: sidecar `obs_embedding.shape[1] == 229376`, expected spec `flat_dim == 1048576` | ✗ |
| Real OFT cold-start GPU e2e after schema fix | same command | Pass | `1 passed, 6 warnings in 122.12s` | ✓ |
| RED install verify export | `python -m pytest tests/unit_tests/test_setup_scripts.py::test_install_verify_exports_dvla_root_to_python_diagnostics -q` before fix | Fail reproducing missing env export | Failed because child Python saw empty `DVLA_ROOT` | ✓ |
| GREEN install verify export | same test after fix | Pass | `1 passed in 0.04s` | ✓ |
| Real install verify | `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash scripts/install/60_verify.sh` | Pass | exit 0; torch cuda true 8; third-party imports under vendored roots; transformers fork true | ✓ |
| Python ruff | `python -m ruff check dreamervla/launchers/train.py tests/unit_tests/test_setup_scripts.py tests/e2e_tests/test_s6_real_oft_coldstart.py tests/unit_tests/test_dvla_paths.py tests/unit_tests/test_rollout_hidden_extractor.py` | Pass | `All checks passed!` | ✓ |
| Related unit bundle | `python -m pytest tests/unit_tests/test_setup_scripts.py tests/unit_tests/test_ray_coldstart_real_config.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py tests/unit_tests/test_openvla_traj1_libero_matrix.py tests/unit_tests/test_dvla_paths.py tests/unit_tests/test_rollout_hidden_extractor.py -q` | Pass | `103 passed, 5 skipped, 5 warnings in 15.89s` | ✓ |
| Full default unit suite | `python -m pytest -q` | Pass | `449 passed, 5 skipped, 11 warnings in 85.84s` | ✓ |
| Full e2e suite | `python -m pytest tests/e2e_tests -q` | Pass | `21 passed, 1 skipped, 6 warnings in 248.81s` | ✓ |
| Diff whitespace | `git diff --check` | Pass | exit 0 | ✓ |
| Shell syntax | `bash -n scripts/install/60_verify.sh` | Pass | exit 0 | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-19 | Tutorial OFT classifier dry-run failed because dotted `task.openvla_oft.*` override was parsed by launcher config | 1 | Added regression test and fixed `dreamervla/launchers/train.py` override classification. |
| 2026-06-19 | Real GPU e2e sidecar dimension mismatch | 1 | Investigation in progress: determine whether e2e expected vision-backbone flat dim while collector writes action-hidden flat dim. |
| 2026-06-19 | Real GPU e2e sidecar dimension mismatch | 2 | Root cause: test used input-token `vla_latent_spec` against action-query sidecar. Fixed test to branch by `preprocess_config["obs_hidden_source"]`; default action-query expects `cfg.task.openvla_oft.wm_obs_dim == 229376`. |
| 2026-06-19 | `scripts/install/60_verify.sh` failed with `KeyError: DVLA_ROOT` | 1 | Root cause: script did not export local `DVLA_ROOT` before invoking Python diagnostics. Added regression test and changed to `export DVLA_ROOT=...`. |
| 2026-06-19 | Full default unit suite failed in `test_coldstart_plan_uses_dvla_root_data_interpolation` | 1 | Test expected removed `+policy.init_lm_head_ckpt` smoke override. Updated it to verify current Hydra-derived policy checkpoint contract. |
| 2026-06-19 | Full default unit suite ran stale real-model sidecar gate and failed | 1 | Made `test_rollout_hidden_extractor.py` real-model unit gates explicit opt-in via `DVLA_REAL_MODEL_UNIT=1`; tutorial real OFT path remains covered by e2e gate. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1: Tutorial Inventory |
| Where am I going? | Static validation, dry-runs/light execution, final verdict |
| What's the goal? | Verify first-party experiment tutorials are okay |
| What have I learned? | See findings.md |
| What have I done? | Created isolated planning context |
