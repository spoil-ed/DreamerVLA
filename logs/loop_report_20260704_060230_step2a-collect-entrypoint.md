# Loop Report: step2a-collect-entrypoint

## 本步目标

执行 SPEC `Step 2 — EGL 三处对齐（R3）` 的第一个可独立验证子步：实证主线 collect 的真实渲染入口，确认 `ColdStartRayCollectRunner` 与 `collect_parallel_rollouts.py:497` 的关系，避免后续 EGL 默认值改错路径。

## 改了哪些文件

- `logs/loop_progress.md`: 将原 `Step 2` 拆为 `Step 2a/2b/2c`，并把 `Step 2a` 标为 `DONE`。
- `logs/loop_report_20260704_060230_step2a-collect-entrypoint.md`: 记录本轮入口追踪、验证命令和结论。

本轮没有修改训练代码、配置默认值或测试文件。

## 验证命令与真实输出摘要

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

plan = build_pipeline_plan(mode='ray', run_root=Path('/tmp/dvla-step2a'), python='python', profile='smoke', ngpu=1)
print('collect_cmd', ' '.join(plan.collect_cmd))
print('has_collect_rollouts_ray', 'experiment=collect_rollouts_ray' in plan.collect_cmd)
print('has_collect_rollouts_onetraj', 'experiment=collect_rollouts_onetraj' in plan.collect_cmd)
print('has_render_backend', [x for x in plan.collect_cmd if x.startswith('render_backend=')])
PY
```

结果：`collect_cmd` 使用 `python -m dreamervla.train experiment=collect_rollouts_ray ...`；`has_collect_rollouts_ray True`；`has_collect_rollouts_onetraj False`；`has_render_backend []`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.config_resolvers import register_dreamervla_resolvers

register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=['experiment=collect_rollouts_ray'])
OmegaConf.resolve(cfg)
print('target', cfg._target_)
print('mode', cfg.mode)
print('env_target', cfg.env.cfg.target)
print('env_render_backend', OmegaConf.select(cfg, 'env.cfg.render_backend', default=None))
PY
```

结果：`target dreamervla.runners.ColdStartRayCollectRunner`；`mode oft`；`env_target dreamervla.envs.train_env:DreamerVLAOnlineTrainEnv`；`env_render_backend None`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=[
        'experiment=collect_rollouts_ray',
        '+env.cfg.render_backend=egl',
        'collect.task_ids=[0]',
        'collect.episodes_per_task=1',
    ])
OmegaConf.resolve(cfg)
plan = ColdStartRayCollectRunner(cfg).build_oft_worker_plan()
print('target', cfg._target_)
print('plan_env_target', plan['env']['target'])
print('plan_render_backend', plan['env'].get('render_backend'))
print('runner_launches_envworker', 'ColdStartRayCollectRunner._build_oft_components -> WorkerGroup(EnvWorker, plan["env"])')
PY
```

结果：`plan_render_backend egl`，说明 `env.cfg.render_backend` 一旦进入 Hydra config，就会随 `build_oft_worker_plan()` 传入 `EnvWorker`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_rollouts_ray_experiment_composes tests/unit_tests/test_ray_coldstart_real_config.py::test_runner_builds_bundle_cfg_from_central_config -q
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ray_launcher_plan_wires_coldstart_outputs_into_cotrain_warmup -q
```

结果：分别为 `2 passed in 1.75s` 和 `1 passed in 1.66s`。

```bash
rg -n "collect_rollouts\(|_build_resume_work_list|WorkerGroup\(EnvWorker|os.environ.setdefault\(\"MUJOCO_GL\"" dreamervla/runners/cold_start_ray_collect_runner.py dreamervla/runners/collect_rollouts_runner.py dreamervla/runners/collect_parallel_rollouts.py
```

结果：`collect_rollouts_runner.py` 调用 `collect_rollouts(...)`；`cold_start_ray_collect_runner.py` 只从 `collect_parallel_rollouts` import `_build_resume_work_list`，OFT path 中 `WorkerGroup(EnvWorker, env_cfg, ...)` 是实际 env runner；`collect_parallel_rollouts.py:497` 的 `MUJOCO_GL=osmesa` 属于 no-Ray collector function。

另外验证了负例：

```bash
env.cfg.render_backend=egl
```

直接 Hydra override 当前失败，报 `Key 'render_backend' is not in struct`，提示使用 `+env.cfg.render_backend=egl`。这说明 `Step 2b` 需要在 Ray collect config 中声明/默认 `env.cfg.render_backend`，不能只依赖普通 override。

## 结论

`DONE`。主线 Ray collect 真实渲染入口已确认：

- launcher `mode=ray` 生成 `experiment=collect_rollouts_ray`；
- `configs/experiment/collect_rollouts_ray.yaml` 的 `_target_` 是 `dreamervla.runners.ColdStartRayCollectRunner`；
- OFT collect path 在 `ColdStartRayCollectRunner._build_oft_components()` 中启动 `WorkerGroup(EnvWorker, plan["env"])`；
- `collect_parallel_rollouts.py:497` 的 `MUJOCO_GL=osmesa` 不控制 Ray collect OFT env 渲染，仅影响 no-Ray `CollectRolloutsRunner.collect_rollouts()` 路径。

## 下一步建议

继续 `Step 2b`：在 Ray collect config/launcher 路径声明并默认传递 `env.cfg.render_backend=egl`，同时调整 cotrain-real/eval 默认 EGL；保留 `render_backend=osmesa` 显式回退与零 GPU EGL 拒绝。

## 残留风险

- 本轮没有改变默认 EGL；GPU 冒烟不适用于本子步。
- 当前 Ray collect 默认 `env.cfg.render_backend` 仍为 `None`，需要 `Step 2b` 修复。
- 当前 worktree 仍有大量本轮前 staged rename 和 modified 文件，本轮未改动它们。
