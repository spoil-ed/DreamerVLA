# Cotrain readiness-gate + egl env-var wiring fix

**Self-contained plan.** Two surgical changes to `dreamervla/runners/online_cotrain_runner.py`, both on
the online-cotrain training path, plus one GPU smoke to verify. No other files. All commands use the
`dreamervla` conda env. Base commit: `8878414` (main).

---

## Change A — readiness-gate (perf, behavior-equivalent)  [APPLIED]

### Problem
`_run_training_bursts(...)` (the shared burst function called once **per env-step** by both the legacy
single-env loop, `online_cotrain_runner.py:685`, and the vectorized `train_hook`, `:890`) computes replay
readiness on **every** env-step:

```python
# online_cotrain_runner.py:719 (pre-change)
_stats, _cov_ready, all_ready = get_replay_task_stats_global(
    replay, task_ids=env_task_ids, min_transitions=knobs["min_replay"],
    min_episodes_per_task=knobs["min_eps"], device=self.device,
    is_dist=knobs["is_dist"], world_size=self._world_size,
)
num_updates = 0
if all_ready and env_step % knobs["train_every"] == 0:   # readiness only USED on the boundary
    num_updates = knobs["updates_per_train"]
```

`get_replay_task_stats_global` scans the replay (O(episodes)) **and does a DDP `all_reduce`** under
`is_dist`. But the result is only consulted when `env_step % train_every == 0` (default `train_every=8`),
so ~7/8 of the calls are pure waste (a per-step scan + collective that changes nothing).

### Fix (applied at `online_cotrain_runner.py:718-730`)
Early-return on non-boundary steps, before the scan/collective:

```python
        # ---- training bursts (lockstep across ranks via global-ready flag)
        # Readiness (and its per-step DDP all_reduce) is only consulted on
        # train_every boundaries, so skip the replay scan + collective on the other
        # steps — identical training cadence, fewer per-step scans/collectives. All
        # ranks gate on the shared env_step, so the all_reduce stays in lockstep.
        if env_step % knobs["train_every"] != 0:
            return False
        _stats, _cov_ready, all_ready = get_replay_task_stats_global( ... )   # unchanged args
        num_updates = knobs["updates_per_train"] if all_ready else 0
```

### Why this is behavior-equivalent
- On non-boundary steps the pre-change code set `num_updates=0` → the `for _ in range(num_updates)` loop
  (`:731`) was empty → fell through to `return False` (`:833`). There is **no per-step code after that
  loop** (verified: `:832 self.global_step += 1` is the loop's last line, `:833 return False`). So
  early-returning `False` on non-boundary steps is identical.
- DDP lockstep: `env_step` is the deterministic loop counter, identical across ranks, so all ranks skip /
  run `get_replay_task_stats_global` (hence its `all_reduce`) on the SAME steps → no collective desync.

### Verify A
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_vec_rollout.py -q` (the fake
  `train_hook` tests don't call `_run_training_bursts`, so they only guard against import/syntax breakage).
- Real behavior is covered by the GPU smoke below (must still reach the same RL updates).

---

## Change B — egl env-var wiring fix (the real cause of the egl SIGABRT)  [TO APPLY]

### Problem
When `online_rollout.render_backend=egl`, `_run_vectorized_cotrain` builds the child env vars and **forces
`PYOPENGL_PLATFORM=egl`**:

```python
# online_cotrain_runner.py:867-873 (current)
env_vars = {
    k: os.environ[k]
    for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DVLA_DATA_ROOT", "LIBERO_CONFIG_PATH")
    if k in os.environ
}
env_vars["MUJOCO_GL"] = render_backend
env_vars["PYOPENGL_PLATFORM"] = render_backend   # <-- forces egl onto PyOpenGL
```

mujoco renders offscreen via its **own** EGL (selected by `MUJOCO_GL=egl`). Forcing PyOpenGL into egl too
(`PYOPENGL_PLATFORM=egl`) conflicts with mujoco's egl context and **aborts `robosuite ... read_pixels`
(SIGABRT)** — observed in a spawned VecRolloutEnv child:
`Fatal Python error: Aborted ... binding_utils.py:171 read_pixels`. The data-collection path
(`collect_parallel_rollouts.py:271-275`) does **not** force `PYOPENGL_PLATFORM` — it only passes through
`os.environ` (default `osmesa`), which is why egl multi-env collection works there. So this forced override
is the bug, not egl itself.

### Fix (at `online_cotrain_runner.py:872-873`)
Set only `MUJOCO_GL`; leave `PYOPENGL_PLATFORM` at the parent's value (`osmesa`, set at module top),
matching the working collection setup:

```python
        env_vars["MUJOCO_GL"] = render_backend
        # Do NOT force PYOPENGL_PLATFORM to egl: mujoco renders via its own EGL
        # (MUJOCO_GL); forcing PyOpenGL to egl too conflicts and aborts robosuite
        # read_pixels (SIGABRT). The collection path leaves it at the parent's
        # osmesa and runs egl multi-env fine — match that.
```

(i.e. delete the `env_vars["PYOPENGL_PLATFORM"] = render_backend` line.)

### Verify B (GPU smoke — the empirical egl test)
Free GPU = 1/2/3 (check `nvidia-smi`). Launch the cotrain with `render_backend=egl` and a parent that does
NOT set `PYOPENGL_PLATFORM=egl`:

```bash
R=/mnt/data/spoil/workspace/DreamerVLA
conda run -n dreamervla env CUDA_VISIBLE_DEVICES=1 DVLA_DATA_ROOT=$R/data \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NCCL_NVLS_ENABLE=0 PYTHONFAULTHANDLER=1 \
  python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden task=openvla_onetraj_coldstart_libero logger=tensorboard \
  training.debug=true \
  offline_warmup.data_dir=$R/data/collected_rollouts/libero_goal_coldstart_smoke/reward \
  offline_warmup.hidden_dir=$R/data/collected_rollouts/libero_goal_coldstart_smoke/hidden \
  offline_warmup.task_id=null 'env.task_ids=[0,1]' \
  online_rollout.debug_episode_horizon=80 online_rollout.debug_total_env_steps=480 \
  online_rollout.debug_min_replay=80 \
  online_rollout.num_envs=4 online_rollout.render_backend=egl
```
- Parent stays osmesa (it never renders); children get `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=osmesa`.
- **PASS** = no `read_pixels` / `Aborted` / SIGABRT, reaches `[online-cotrain] vectorized rollout: 4 envs,
  render_backend=egl`, runs the rollout faster than the osmesa baseline (osmesa was ~6.4 env/s; egl should
  be materially higher), reaches the RL bursts, exits clean. This also re-validates Change A (same RL path).
- **FAIL** (still SIGABRTs) ⇒ egl on this host is a deeper driver issue, not the env-var override; then keep
  `render_backend=osmesa` (proven working, 4-env parallel) and report.

### Note (collection)
The collection CODE is already correct (no forced `PYOPENGL_PLATFORM`). To collect with egl, just launch
with `MUJOCO_GL=egl` and **do not** set `PYOPENGL_PLATFORM=egl` (the earlier collection SIGABRT was caused
by my launch setting `PYOPENGL_PLATFORM=egl`, not by the code).

---

## Commits
1. `perf(cotrain): gate online readiness check + DDP all_reduce to train_every boundaries` (Change A).
2. `fix(cotrain): don't force PYOPENGL_PLATFORM=egl on vectorized rollout children` (Change B).
Conventional + `--signoff`, no `===`/`/` in subjects. Not pushed.

## Out of scope
Other perf-audit items (Wave 3 prompt-cache / Q11 / Q2+Q6, and Phase 3 W6) — tracked in
`docs/plans/2026-06-23-perf-audit-execution-roadmap.md`, done in their own plans/agents.
