# Step 2a 报告：LIBERO render regime 单一 helper

## 本步目标

按 R3 补充要求，先建立无 GPU 可测的底层 helper：
`dreamervla/utils/egl_device.py::apply_libero_render_regime(backend, shard_id, gpu_pool)`。
本轮只覆盖 helper 行为，不改 collect / cotrain-real / eval 三处调用点，也不切默认值。

## 改动文件

- `dreamervla/utils/egl_device.py`：新增 `apply_libero_render_regime()`，统一处理 LIBERO 的
  `egl` / `osmesa` 后端环境变量；EGL 按 `shard_id % len(gpu_pool)` 选择
  `MUJOCO_EGL_DEVICE_ID`，空池抛现有零 GPU EGL 错误文本；osmesa 在同一 helper 内设置
  `MUJOCO_GL` / `PYOPENGL_PLATFORM` 并清理残留 `MUJOCO_EGL_DEVICE_ID`。
- `tests/unit_tests/test_egl_device.py`：新增无 GPU 单测，覆盖 EGL shard 选卡、osmesa fallback、
  空 GPU 池拒绝 EGL、非法 backend 拒绝。
- `logs/loop_progress.md`：按用户补充修正事实源路径为
  `docs/superpowers/plans/2026-07-04-mainline-deprecation-egl-align.md`，并把 R3 拆分更新为
  helper / 三处接线 / 默认切换。

## 验证命令与输出摘要

1. RED：
   `python -m pytest tests/unit_tests/test_egl_device.py -q`
   - 输出摘要：4 个测试失败，失败原因均为
     `AttributeError: module 'dreamervla.utils.egl_device' has no attribute 'apply_libero_render_regime'`。

2. GREEN：
   `python -m pytest tests/unit_tests/test_egl_device.py -q`
   - 输出摘要：`4 passed in 1.44s`。

3. base env 回归尝试：
   `python -m pytest tests/unit_tests/test_egl_device.py tests/unit_tests/test_online_egl_venv.py tests/unit_tests/test_cotrain_render_backend.py -q`
   - 输出摘要：收集失败，base Python 3.13 环境缺 `gym` / `transformers`，属于 loop 提示中的伪失败。

4. dreamervla env 回归：
   `conda run -n dreamervla python -m pytest tests/unit_tests/test_egl_device.py tests/unit_tests/test_online_egl_venv.py tests/unit_tests/test_cotrain_render_backend.py -q`
   - 输出摘要：`12 passed, 2 warnings in 7.28s`。

5. lint：
   `conda run -n dreamervla ruff check dreamervla/utils/egl_device.py tests/unit_tests/test_egl_device.py`
   - 输出摘要：`All checks passed!`

## 结论

DONE。R3 的单一底层 helper 已具备无 GPU 单测覆盖，后续三处 LIBERO render 入口可以只调用该
helper。

## 下一步建议

继续 Step 2b：把 collect / cotrain-real / eval 三处 LIBERO env 构造改成在各自 env-worker
子进程入口最早处调用 `apply_libero_render_regime()`，传入正确的 `shard_id` 和 config 驱动的
`gpu_pool`，并移除各自分散的 `os.environ.setdefault` / backend 逻辑。

## 残留风险

- 本轮未触碰真实 LIBERO/MuJoCo 初始化路径；端到端 EGL 防 SIGABRT 仍需 Step 2b 接线后在有 GPU
  环境验证。
- 当前 helper 复用 `apply_egl_device_regime()`，会同步设置 `CUDA_VISIBLE_DEVICES` 与
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`；这与既有 EGL helper 行为一致，Step 2b 接线时需确认
  不与 Ray placement 的 per-worker env 冲突。
