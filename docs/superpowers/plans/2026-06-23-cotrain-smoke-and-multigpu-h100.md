# Cotrain full-process smoke + multi-GPU H100 (noray & ray) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Ship a low-cost full-process smoke for the OFT action-hidden online-cotrain pipeline (warmup → egl rollout → ≥1 wmpo_outcome/PPO update), flip the mainline cotrain defaults (logger → tensorboard+wandb offline; env layer → egl multi-env), and make BOTH the noray and ray coldstart→warmup→cotrain paths usable on a single-node 8× H100 box (cotrain already multi-GPU via torchrun DDP; this adds multi-GPU **collection**).

**Architecture:** Config-first. The smoke is a thin experiment over `online_cotrain_pipeline_oft_action_hidden` differing from production ONLY by step counts (`training.debug=true`). Defaults flip in the pipeline base config + the runner code default. noray collection becomes multi-GPU by wrapping the launcher's collect command in `torch.distributed.run` (the collect runner already shards work by torchrun rank and binds `gpu_id=local_rank`). ray collection becomes multi-GPU by launching N inference workers over a GPU range and routing each env to a stable owner worker (`env_id % N`); N=1 is byte-identical to today.

**Tech stack:** Hydra configs, PyTorch DDP/torchrun, Ray WorkerGroup + PackedPlacementStrategy, LIBERO/Mujoco egl.

**Verification reality:** The 8× H100 box is currently saturated; the author runs the real multi-GPU jobs on another machine. Here we verify statically only: Hydra `--cfg job` composition, launcher `dry_run` (prints the exact torchrun commands), `bash -n`, unit tests, ruff.

---

## File structure

- Create `configs/experiment/online_cotrain_pipeline_oft_action_hidden_smoke.yaml` — the full-process smoke recipe.
- Modify `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` — flip mainline logger default.
- Modify `dreamervla/runners/online_cotrain_runner.py` — flip `num_envs` code default 1→4.
- Modify `configs/experiment/online_cotrain_oft_backbone_latent.yaml` — guard `num_envs: 1`.
- Modify `dreamervla/launchers/coldstart_warmup_cotrain.py` — wrap noray collect in torchrun when `ngpu>1`.
- Modify `dreamervla/runners/cold_start_ray_collect_runner.py` — multi-GPU ray collection inference (N workers, env→owner routing).
- Modify `configs/scripts/coldstart_warmup_cotrain.yaml` — `collect.num_inference_workers` knob + control mapping + multi_gpu ray profile.
- Modify `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md` — 8× H100 launch (noray + ray, full + smoke).
- Create `tests/unit_tests/test_ray_collect_inference_sharding.py` — env→owner partition helper test.

---

## Task 1: Full-process smoke config

**Files:** Create `configs/experiment/online_cotrain_pipeline_oft_action_hidden_smoke.yaml`

The smoke composes the production pipeline and differs ONLY by step counts (`training.debug=true` swaps in the runner's `debug_*` values: wm/classifier warmup 2/2, total_env_steps 160, min_replay 48, max_train_updates 4, episode_horizon 50, batch 2, ppo_rollouts 2). `num_envs` stays at the production 4 (inherited); logger is inherited from the flipped base (tb+wandb offline). `checkpoint_every` is lowered so a checkpoint is exercised inside the smoke (a step count).

```yaml
# @package _global_
# Full-process low-cost smoke for the OFT action-hidden online-cotrain pipeline.
# Differs from `online_cotrain_pipeline_oft_action_hidden` ONLY by step counts
# (training.debug=true -> the runner's debug_* swap). Spans every phase:
# offline WM+classifier warmup -> vectorized egl rollout (num_envs=4, == production)
# -> >=1 wmpo_outcome (PPO-style) actor update + inline classifier -> checkpoint.
# Requires the real OFT ckpt + action-hidden sidecar + LIBERO assets + a GPU.
#   python -m dreamervla.train experiment=online_cotrain_pipeline_oft_action_hidden_smoke task=...
defaults:
  - online_cotrain_pipeline_oft_action_hidden
  - _self_

training:
  debug: true   # ONLY difference vs production: tiny debug_* step counts
  out_dir: ${oc.env:DVLA_DATA_ROOT,${oc.env:DVLA_ROOT,.}/data}/outputs/dreamervla/online_cotrain_pipeline_smoke/${now:%Y%m%d_%H%M%S}
  checkpoint_every: 50   # exercise a checkpoint write within the smoke
```

- [ ] Write the file as above.
- [ ] Verify composition (Task 9).

## Task 2: Flip mainline logger default

**Files:** Modify `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` (runner.logger block, ~lines 283-287)

```yaml
runner:
  logger:
    logger_backends: [tensorboard, wandb]
    project_name: dreamervla
    log_path: ${training.out_dir}/log
    wandb_mode: offline
    wandb_proxy: null
```

Both the production pipeline and the smoke inherit this. `runner.logger.wandb_mode` is consumed by `dreamervla/utils/metric_logger.py:126`.

## Task 3: Flip num_envs code default 1 → 4 (RLinf-aligned env layer)

**Files:** Modify `dreamervla/runners/online_cotrain_runner.py:581`

```python
# RLinf-aligned default: vectorized egl multi-env rollout. Was 1 (legacy single-env
# osmesa). 4 matches the shipped pipeline config; bare/ad-hoc runs no longer fall back
# to single-env osmesa. backbone_latent must override to 1 (validate_rollout_cfg).
num_envs = int(OmegaConf.select(oc, "num_envs", default=4))
```

## Task 4: Guard backbone_latent (must stay single-env)

**Files:** Modify `configs/experiment/online_cotrain_oft_backbone_latent.yaml`

Add an explicit single-env override (backbone_latent fails `validate_rollout_cfg` for num_envs>1):

```yaml
online_rollout:
  sequence_length: ${task.openvla_oft.wm_sequence_length}
  num_envs: 1            # backbone_latent requires single-env (validate_rollout_cfg)
  render_backend: osmesa
```

## Task 5: noray collection multi-GPU (wrap collect in torchrun)

**Files:** Modify `dreamervla/launchers/coldstart_warmup_cotrain.py` (collect_cmd build, ~lines 205-227)

The collect runner already shards by torchrun rank and binds `gpu_id=local_rank`
(`collect_parallel_rollouts.py:357-358`, `collect_rollouts_runner.py:81-92`). So for
mode=noray + ngpu>1, wrap the collect command in `torch.distributed.run` exactly like
the cotrain command. Ray mode keeps its own Ray fan-out (no torchrun).

```python
collect_launch = [python_cmd, "-m"]
if selected_mode == "noray" and distributed and selected_ngpu > 1:
    collect_launch += [
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={selected_ngpu}",
        f"--master_port={selected_master_port}",
        "-m",
    ]
collect_cmd = [
    *collect_launch,
    "dreamervla.train",
    *_render_overrides(mode_cfg["collect"], context),
    *_render_overrides(collect_profile_cfg, context),
]
```

(The cotrain torchrun wrap already exists at lines 229-237; reuse `selected_master_port`. If the same master_port collides when both stages run back-to-back it is fine — they run sequentially.)

## Task 6: ray collection multi-GPU (N inference workers + env→owner routing)

**Files:** Modify `dreamervla/runners/cold_start_ray_collect_runner.py`

Today inference is one worker on one GPU (`PackedPlacementStrategy(gpu_id, gpu_id)`,
line 234-236) and the loop calls `infer.forward_batch(obs_batch, env_ids)` on it
(line 316). Make inference data-parallel across GPUs with a stable env→owner map.

**6a. Launch N inference workers (setup, ~line 234-237):**

```python
gpu_id = int(collect_cfg.get("gpu_id", 0))
num_infer = max(1, int(collect_cfg.get("num_inference_workers", 1)))
end_gpu = gpu_id + num_infer - 1
infer_group = WorkerGroup(
    RolloutInferenceWorker, plan["inference"], {}, num_envs=num_envs
).launch(cluster, PackedPlacementStrategy(gpu_id, end_gpu, num_gpus_per_worker=1))
# store for the loop
groups_extra = {"num_infer": num_infer}
```

Return `num_infer` in the dict (add `"num_infer": num_infer`).

**6b. Pure helper (top of module) — env_ids grouped by owner worker:**

```python
def _shard_env_ids_by_worker(env_ids: list[int], num_workers: int) -> dict[int, list[int]]:
    """Stable partition: env_id -> owner worker (env_id % num_workers).
    Returns {worker_rank: [env_id, ...]} preserving input order within each worker."""
    groups: dict[int, list[int]] = {w: [] for w in range(num_workers)}
    for env_id in env_ids:
        groups[int(env_id) % num_workers].append(int(env_id))
    return {w: ids for w, ids in groups.items() if ids}
```

**6c. Route forward_batch / reset_states in `_run_loop` (lines 305-347) — replace the single-worker calls:**

```python
num_infer = int(groups.get("num_infer", 1))
...
# obs_batch is in env_ids order; map each env's obs to its owner worker
owner = {int(e): int(e) % num_infer for e in env_ids}
shards = _shard_env_ids_by_worker(env_ids, num_infer)
obs_by_env = dict(zip(env_ids, obs_batch, strict=True))
infer_calls = {
    w: infer.execute_on(w).forward_batch([obs_by_env[e] for e in ids], ids)
    for w, ids in shards.items()
}
out_by_env: dict[int, tuple] = {}
for w, ids in shards.items():
    out = wait_result(infer_calls[w])[0]
    for e, a, h in zip(ids, out["actions"], out["obs_embedding"], strict=True):
        out_by_env[e] = (a, h)
step_calls = [
    envs.execute_on(e).step(out_by_env[e][0], out_by_env[e][1]) for e in env_ids
]
...
# reset_states per owner worker
if done_envs:
    reset_by_worker = _shard_env_ids_by_worker(done_envs, num_infer)
    wait_results([infer.execute_on(w).reset_states(ids) for w, ids in reset_by_worker.items()])
```

Apply the same routing in `_run_loop_overlap` if it duplicates the single-worker calls.

**Gate:** with `num_inference_workers=1`, `shards == {0: env_ids}` and behavior is identical to today.

**Caveat (document in commit + tutorial):** the >1 path is logic-verified + unit-tested
here but NOT GPU-verified; validate on the 8× H100 box before a long collection.

## Task 7: ray collect knob + multi_gpu profile

**Files:** Modify `configs/scripts/coldstart_warmup_cotrain.yaml`

- Add control knob under `collect:` → `num_inference_workers: null`.
- Add mapping under `control_overrides.collect.ray:` → `num_inference_workers: collect.num_inference_workers`.
- In `profiles.multi_gpu.collect.ray`, add `- collect.num_inference_workers=4`.
- (noray collect multi-GPU needs no profile change — it is driven by `ngpu`.)

Also `scripts/start_ray.sh` defaults `--num-gpus` to `RAY_NUM_GPUS:-0`; the tutorial
must `export RAY_NUM_GPUS=8` for the ray multi-GPU run.

## Task 8: Tutorial — 8× H100 launch (noray + ray, full + smoke)

**Files:** Modify `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md`

Add a "Multi-GPU (8× H100, single node)" subsection after §1 with:

```bash
# Env (both modes): vectorized egl rollout + DDP cotrain
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_NVLS_ENABLE=0

# no-Ray full: collect (8-rank torchrun) + warmup + cotrain (8-rank DDP)
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=8 profile=multi_gpu

# Ray full: ray collect (4 inference GPUs) + cotrain (8-rank DDP)
export RAY_NUM_GPUS=8
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=8 profile=multi_gpu \
  collect.num_inference_workers=4

# Smoke (either mode): full process, tiny steps
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=8 profile=multi_gpu debug=true
# or the single-call cotrain-only smoke (assets already collected):
python -m dreamervla.train experiment=online_cotrain_pipeline_oft_action_hidden_smoke task=openvla_onetraj_coldstart_libero
```

Note in the render-backend callout that the multi-GPU default is now egl multi-env
(num_envs=4); fall back to `online_rollout.num_envs=1` + osmesa only if egl aborts.

## Task 9: Static verification (no GPU)

- [ ] `conda run -n dreamervla python -m dreamervla.train experiment=online_cotrain_pipeline_oft_action_hidden_smoke task=openvla_onetraj_coldstart_libero --cfg job` → resolves, debug=true, num_envs=4, logger=[tensorboard,wandb] wandb_mode=offline.
- [ ] Same `--cfg job` for `online_cotrain_pipeline_oft_action_hidden` (logger flipped) and `online_cotrain_oft_backbone_latent` (num_envs=1, no validate error at compose).
- [ ] `python -m dreamervla.launchers.coldstart_warmup_cotrain mode=noray ngpu=8 profile=multi_gpu task=goal dry_run=true` → collect AND cotrain commands both wrapped in `torch.distributed.run --nproc-per-node=8`.
- [ ] `... mode=ray ngpu=8 profile=multi_gpu collect.num_inference_workers=4 dry_run=true` → collect is Ray (no torchrun), cotrain wrapped in torchrun; collect cmd carries `collect.num_inference_workers=4`.
- [ ] `bash -n scripts/e2e_coldstart_warmup_cotrain_noray.sh scripts/e2e_coldstart_warmup_cotrain_ray.sh scripts/start_ray.sh`.
- [ ] `conda run -n dreamervla python -m pytest tests/unit_tests/test_ray_collect_inference_sharding.py tests/unit_tests/test_vec_rollout_env.py -q`.
- [ ] `conda run -n dreamervla ruff check dreamervla tests` on changed files.

## Task 10: Unit test for env→owner partition

**Files:** Create `tests/unit_tests/test_ray_collect_inference_sharding.py`

```python
from dreamervla.runners.cold_start_ray_collect_runner import _shard_env_ids_by_worker


def test_single_worker_owns_all_in_order():
    assert _shard_env_ids_by_worker([0, 1, 2, 3], 1) == {0: [0, 1, 2, 3]}


def test_round_robin_partition_stable_and_complete():
    shards = _shard_env_ids_by_worker([0, 1, 2, 3, 4, 5], 3)
    assert shards == {0: [0, 3], 1: [1, 4], 2: [2, 5]}
    # every env_id appears exactly once
    flat = sorted(e for ids in shards.values() for e in ids)
    assert flat == [0, 1, 2, 3, 4, 5]


def test_empty_workers_omitted():
    assert _shard_env_ids_by_worker([0, 2], 2) == {0: [0, 2]}
```

## Task 11: Commit + push

- [ ] `git add -A && git commit --signoff -m "feat: full-process cotrain smoke + multi-GPU H100 (noray & ray) collect"` (conventional subject, no `===`/`/`).
- [ ] `git push -u origin feat/cotrain-smoke-and-multigpu-h100`.
- [ ] Report what is static-verified vs pending GPU validation (ray collect >1 inference workers; any real 8-GPU run).
</content>
</invoke>
