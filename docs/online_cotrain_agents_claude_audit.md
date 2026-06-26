# Online Cotrain AGENTS/CLAUDE Compliance Audit

Scope:
`online_cotrain_runner.py`, `online_cotrain_ray_runner.py`, `learner_worker.py`,
`online_replay.py`, `offline_seed.py`, `algorithms/ppo/outcome.py`, and relevant Hydra
configs.

## Must Fix Before Training

Re-audit on 2026-06-25 found implementation gaps that must be fixed before
training claims:

1. Warmup config used fixed 1200-step knobs but did not enable replay coverage
   semantics by default. Fixed by setting `training.warmup_replay_epochs=1`
   with `training.warmup_replay_max_steps=1200`, and by resolving epoch-derived
   steps through one runner helper.
2. Task-conditioning validation existed, but classifier warmup/update did not
   forward replay `task_ids` to task-aware classifiers. Fixed in the shared
   classifier update step.
3. WM hidden reconstruction/cosine metrics were emitted by the algorithm but
   dropped by sync/Ray learner loops. Fixed through the shared
   `namespaced_world_model_metrics` mapper.
4. Ray rollout metrics still exposed duplicate `rollout/current_success_rate`
   and `rollout/avg_success_rate`. Fixed to match the final cumulative/recent
   metric set.

No remaining CPU-testable must-fix item is open in this audit. Gate 8 remains
GPU-only validation and must run separately on the approved GPUs.

## Should Fix Soon

1. Runtime config mutation for backbone latent env hidden source
   File: `dreamervla/runners/online_cotrain_runner.py`
   Classification: `should-fix-soon`
   Detail: the runner still sets `env.obs_hidden_source` when `latent_type=backbone_latent`.
   This is not `_target_` mutation and does not affect the default action-hidden route, but
   the contract parameter should eventually live directly in the relevant Hydra recipe.

2. Training-loop status uses `print` in some warmup/online paths
   Files: `online_cotrain_runner.py`, `online_cotrain_pipeline_runner.py`
   Classification: `should-fix-soon`
   Detail: warmup progress and main update metrics now use namespaced logger calls, but
   some human status messages remain as `print`. They are operational progress messages,
   not scalar metrics. A later cleanup can route them through the runner logger/console
   consistently.

3. Ray runner OFT inference plan is built through collect-runner helper
   File: `dreamervla/runners/online_cotrain_ray_runner.py`
   Classification: `should-fix-soon`
   Detail: this avoids hand-authored OFT YAML duplication, but the helper lives on a
   concrete collect runner class. Extracting a neutral OFT rollout-plan builder would better
   preserve role-based naming and decoupling.

## Acceptable Boundaries

1. Offline seed assumes collected LIBERO-style HDF5 schema
   File: `dreamervla/runners/offline_seed.py`
   Classification: `acceptable-boundary`
   Rationale: AGENTS.md states new envs beyond LIBERO are not stable; this loader is an
   external data-boundary utility for collected reward/hidden HDF5.

2. Ray LearnerWorker uses plain target dictionaries rather than Hydra `instantiate`
   File: `dreamervla/workers/actor/learner_worker.py`
   Classification: `acceptable-boundary`
   Rationale: Ray worker configs are serialized plain dicts with `target`/`_target_`;
   `_build_from_cfg` preserves target-based construction across process boundaries.

3. LUMOS outcome route is chunk-specific
   File: `dreamervla/algorithms/ppo/outcome.py`
   Classification: `acceptable-boundary`
   Rationale: the route is selected through `algorithm.update_type` registry and validates
   that online cotrain uses the chunk-world-model actor update route.

4. Replay source ids are numeric diagnostics
   File: `dreamervla/runners/online_replay.py`
   Classification: `acceptable-boundary`
   Rationale: source remains episode-level metadata, avoiding new step-level coupling.

## Passed Checks

1. Runner/worker model components are selected through Hydra targets, serialized target
   dictionaries, registries, or narrow capability flags.
2. No `_target_` mutation was added.
3. No new `if model == "openvla"` generic training-loop branch was added.
4. Real replay and imagined rollout memory remain separate.
5. External hidden dimensions remain config/sidecar-driven; internal WM dimensions remain
   model config choices.
6. Optional task conditioning is config-disabled by default and fails fast if enabled with
   unsupported implementations.
7. Ray remains an explicit backend route, not the default topology.
8. Rollout metrics now distinguish real `rollout/*` success from imagined `rl/*` and
   `LUMOS/*` diagnostics.

## CLAUDE.md Recheck

1. AGENTS.md remains the authoritative architecture source; this audit found no
   cotrain-specific rule that conflicts with it.
2. The grouped Hydra route remains the entry point. The changes add/adjust cohesive
   config groups and experiment composition, not a new top-level training route.
3. Ray cotrain remains an explicit optional backend, while the default design stays on the
   single-machine Runner pattern.
4. Actor-update behavior remains selected through the existing algorithm route; no new
   generic training-loop dispatch branch was added for a concrete model name.
5. Implementation code stays under `dreamervla/`; shell/launcher changes remain thin
   command composition.
6. The new behavior is covered by focused unit tests for replay sampling, chunk rollout,
   warmup cadence and cap, LUMOS signal filtering, Ray learner metric forwarding, and
   task conditioning validation/forwarding.
