# Loop Report: step1-r2-baseline-contract

## 本步目标

执行 SPEC `Step 1 — 冻结主线契约（R2）`：确认主线双写点 `real=32 / imagine=256 / step=512`，并在 `dreamervla/config.py` 增加偏离主线 baseline 时“告警但放行”的早校验。

## 改了哪些文件

- `dreamervla/config.py`: 新增 `_warn_manual_cotrain_baseline_overrides()`，仅对 `ManualCotrainRayRunner` 中的 `manual_cotrain.real_rollout_target_trajectories`、`manual_cotrain.wm_rollout_target_trajectories`、`manual_cotrain.max_steps_per_rollout_epoch` 偏离 32/256/512 时发 `UserWarning`，不改变既有 hard validation。
- `tests/unit_tests/test_manual_cotrain_config_validation.py`: 新增 R2 baseline override warning 测试；将本文件 `_cfg()` 默认 `max_steps_per_rollout_epoch` 调整为主线 baseline 512，避免默认测试都触发 R2 warning。
- `logs/loop_progress.md`: 将 `Step 1` 从 `TODO`/`DOING` 推进到 `DONE`。
- `logs/loop_report_20260704_055731_step1-r2-baseline-contract.md`: 记录本轮目标、改动、验证和结论。

未修改本轮前已存在的 YAML 在途 diff；本轮只验证它们当前值符合 R2 双写点。

## 验证命令与真实输出摘要

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_config_validation.py::test_manual_cotrain_warns_when_baseline_rollout_budget_is_overridden -q
```

红灯：实现前失败，3 个 case 均为 `Failed: DID NOT WARN`。

绿灯：实现后 `3 passed, 2 warnings in 4.73s`，只剩环境级 SWIG deprecation warnings。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_config_validation.py -q
```

结果：`37 passed, 6 warnings in 4.80s`。其中 4 个 `UserWarning` 是测试里显式使用非 baseline rollout budget 的可见告警，符合“覆盖时告警但放行”。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers

register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=['experiment=openvla_onetraj_libero_cotrain_ray'])
OmegaConf.resolve(cfg)
validate_cfg(cfg)
print('mainline', cfg.manual_cotrain.real_rollout_target_trajectories, cfg.manual_cotrain.wm_rollout_target_trajectories, cfg.manual_cotrain.max_steps_per_rollout_epoch)
PY
```

结果：`mainline 32 256 512`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -W always::UserWarning - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers

register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=[
        'experiment=openvla_onetraj_libero_cotrain_ray',
        'manual_cotrain.real_rollout_target_trajectories=8',
        'manual_cotrain.wm_rollout_target_trajectories=128',
        'manual_cotrain.max_steps_per_rollout_epoch=64',
    ])
OmegaConf.resolve(cfg)
validate_cfg(cfg)
print('override', cfg.manual_cotrain.real_rollout_target_trajectories, cfg.manual_cotrain.wm_rollout_target_trajectories, cfg.manual_cotrain.max_steps_per_rollout_epoch)
PY
```

结果：打印 3 条 `UserWarning`，分别对应 real 32、wm 256、step 512 baseline 被覆盖；最后输出 `override 8 128 64`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -W always::UserWarning - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers

register_dreamervla_resolvers()
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=['experiment=manual_cotrain_ray_tiny'])
OmegaConf.resolve(cfg)
validate_cfg(cfg)
print('tiny', cfg.manual_cotrain.real_rollout_target_trajectories, cfg.manual_cotrain.wm_rollout_target_trajectories, cfg.manual_cotrain.max_steps_per_rollout_epoch)
PY
```

结果：`manual_cotrain.max_steps_per_rollout_epoch` 覆盖 baseline 512 的 `UserWarning` 可见；最后输出 `tiny None None 2`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python - <<'PY'
from omegaconf import OmegaConf
ray = OmegaConf.load('configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml')
script = OmegaConf.load('configs/scripts/coldstart_warmup_cotrain.yaml')
print('ray', ray.manual_cotrain.real_rollout_target_trajectories, ray.manual_cotrain.wm_rollout_target_trajectories, ray.manual_cotrain.max_steps_per_rollout_epoch)
profile = script.profiles.multi_gpu
print('multi_gpu', profile.ray_online_real_rollout_target_trajectories, profile.ray_online_wm_rollout_target_trajectories, profile.ray_online_max_steps_per_rollout_epoch)
PY
```

结果：`ray 32 256 512` 和 `multi_gpu 32 256 512`。

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m ruff check dreamervla/config.py tests/unit_tests/test_manual_cotrain_config_validation.py
/home/user01/miniconda3/envs/dreamervla/bin/python -m py_compile dreamervla/config.py tests/unit_tests/test_manual_cotrain_config_validation.py
git diff --check -- dreamervla/config.py tests/unit_tests/test_manual_cotrain_config_validation.py logs/loop_progress.md
```

结果：ruff `All checks passed!`；`py_compile` 和 `git diff --check` 均无输出。

## 结论

`DONE`。R2 当前状态满足：主线 compose 三值为 32/256/512；script `multi_gpu` profile 双写点也是 32/256/512；偏离 baseline 时发 `UserWarning` 并放行 tiny/smoke。

## 下一步建议

继续 `Step 2 — EGL 三处对齐（R3）`：先实证主线 collect 真实渲染入口，再调整 collect/cotrain-real/eval 默认 EGL 与 per-worker device binding。

## 残留风险

- 当前 worktree 仍有大量本轮之前的 staged rename 和 modified 文件；本轮只应提交本轮新增/修改的 hunks。
- `manual_cotrain_ray_tiny` 当前没有显式设置 real/wm target，只有 `max_steps_per_rollout_epoch=2` 触发 R2 warning；这是 smoke 覆盖，符合 SPEC 放行策略。
