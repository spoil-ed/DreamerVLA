# Step 2b 报告：三处 LIBERO render 入口接入单一 helper

## 本步目标

按 R3 补充要求，把 collect / cotrain-real / post-step eval 三处 LIBERO render 后端选择接到
`dreamervla.utils.egl_device.apply_libero_render_regime()`，避免三处分别写
`MUJOCO_GL` / `PYOPENGL_PLATFORM` / `MUJOCO_EGL_DEVICE_ID` 逻辑。本轮不切默认值，默认 EGL
留给 Step 2c。

## 改动文件

- `dreamervla/utils/egl_device.py`：让 `apply_libero_render_regime()` 在 `gpu_pool` 为空时可从
  worker 已继承的 `MUJOCO_EGL_DEVICE_ID` / `CUDA_VISIBLE_DEVICES` 派生 EGL 池；EGL/osmesa
  环境变量仍只在该 helper 内设置。
- `dreamervla/workers/env/env_worker.py`：Ray collect 的 spawned child 在 `_build_env_from_cfg()`
  之前调用 `apply_libero_render_regime()`；in-process osmesa fallback 也改为调用同一 helper。
- `dreamervla/workers/env/trajectory_env_worker.py`：manual cotrain real env 在 Ray actor 进程内、
  build real LIBERO env 前调用同一 helper；WM env 不走该 LIBERO render path。
- `dreamervla/launchers/coldstart_warmup_cotrain.py`：post-step eval subprocess env 通过同一 helper
  生成，保留既有 `eval.egl_device_id` 优先、否则取 `eval.gpus` 最后一张卡的行为。
- `tests/unit_tests/test_libero_render_regime_wiring.py`：新增无 GPU wiring 单测，断言 collect
  spawned child、collect in-process fallback、manual real env、post-step eval env 都先经 helper。
- `logs/loop_progress.md`：Step 2b 标为 DONE，下一步转 Step 2c。

## 验证命令与输出摘要

1. RED：
   `conda run -n dreamervla python -m pytest tests/unit_tests/test_libero_render_regime_wiring.py -q`
   - 输出摘要：4 个测试失败，均显示 helper call 缺失，env build / eval env 仍走旧逻辑。

2. GREEN：
   `conda run -n dreamervla python -m pytest tests/unit_tests/test_libero_render_regime_wiring.py -q`
   - 输出摘要：`4 passed in 1.30s`。

3. focused regression：
   `conda run -n dreamervla python -m pytest tests/unit_tests/test_egl_device.py tests/unit_tests/test_libero_render_regime_wiring.py tests/unit_tests/test_env_worker_world_model_sync.py tests/unit_tests/test_env_worker_spawn_recovery.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_eval_runs_segmented_post_step_libero_eval_and_writes_trend_summary tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_post_step_eval_egl_device_defaults_to_last_eval_gpu_for_split_render -q`
   - 输出摘要：`17 passed in 1.72s`。

4. lint：
   `conda run -n dreamervla ruff check dreamervla/utils/egl_device.py dreamervla/workers/env/env_worker.py dreamervla/workers/env/trajectory_env_worker.py dreamervla/launchers/coldstart_warmup_cotrain.py tests/unit_tests/test_libero_render_regime_wiring.py`
   - 输出摘要：`All checks passed!`

5. 静态确认：
   `rg -n "os\\.environ\\.setdefault\\(\\\"MUJOCO_GL\\\"|os\\.environ\\[\\\"MUJOCO_GL\\\"\\]|os\\.environ\\.setdefault\\(\\\"PYOPENGL_PLATFORM\\\"|apply_libero_render_regime" dreamervla/workers/env/env_worker.py dreamervla/workers/env/trajectory_env_worker.py dreamervla/launchers/coldstart_warmup_cotrain.py dreamervla/utils/egl_device.py -S`
   - 输出摘要：三处入口只出现 `apply_libero_render_regime()` 调用；直接 `MUJOCO_GL` 写入只保留在
     `dreamervla/utils/egl_device.py`。

## 结论

DONE。collect / cotrain-real / post-step eval 的 LIBERO render 后端选择已收敛到单一 helper。

## 下一步建议

继续 Step 2c：把 collect / cotrain-real / eval 三处 config 级 `render_backend` 默认切到 `egl`，
保留 `render_backend=osmesa` 显式回退，并验证零 GPU + EGL 仍拒绝。

## 残留风险

- 未做真实 GPU EGL 端到端冒烟；按 loop 规则标为后续 GPU-GATED 验证。
- `tests/unit_tests/test_trajectory_env_worker.py::test_real_env_worker_builds_egl_slots_in_spawn_children`
  仍失败于当前 in-flight spawn-slot 设计不完整。该测试在本轮前已与当前源码不一致，本轮未把更大的
  real-env spawn 迁移并入 Step 2b，以免扩大原子步骤。
