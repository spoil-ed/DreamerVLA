# Step 2c Report: EGL 默认切换

## 本步目标

将 collect / cotrain-real / eval 三处 config 级 `render_backend` 默认切到 `egl`，保留显式 `render_backend=osmesa` 回退与零 GPU 拒绝。

## 改动文件

- `configs/scripts/coldstart_warmup_cotrain.yaml`：脚本级 `render_backend` 默认改为 `egl`，Ray collect 命令显式传 `env.cfg.render_backend={render_backend}`，multi_gpu 不再强制 `manual_cotrain.real_render_backend=osmesa`。
- `configs/experiment/collect_rollouts_ray.yaml`：collect env 默认声明 `render_backend: egl`。
- `configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml`：manual cotrain 主 route 顶层默认 `render_backend: egl`。
- `configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml`：Ray base 默认文档值同步为 `egl`。
- `dreamervla/runners/cold_start_ray_collect_runner.py`：新增 collect EGL render pool 兜底；当 env cfg 未显式给 `gpu_pool/render_devices/egl_device_pool` 时，从 `cluster.num_gpus` 派生 `render_devices=[0..N-1]`，N=0 复用既有 `_ZERO_GPU_EGL_ERROR`。
- `tests/unit_tests/test_mainline_egl_defaults.py`：新增 Step 2c 默认 EGL、显式 osmesa fallback、零 GPU EGL 拒绝、Hydra compose 默认值测试。
- `tests/unit_tests/test_ray_coldstart_real_config.py`：新增 collect EGL render pool regression 测试，覆盖默认派生与显式 pool 保留。
- `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`：更新 launcher 断言以区分默认 EGL placement 与显式 osmesa fallback。
- `logs/loop_progress.md`：Step 2c 标为 `BLOCKED`。

## 验证命令与输出摘要

- RED：`conda run -n dreamervla python -m pytest tests/unit_tests/test_mainline_egl_defaults.py -q`
  - 结果：6 failed，失败点为旧默认仍是 osmesa、collect 未透传 `env.cfg.render_backend`、零 GPU 默认未拒绝。
- RED：`conda run -n dreamervla python -m pytest tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_egl_render_pool_defaults_to_cluster_gpus tests/unit_tests/test_ray_coldstart_real_config.py::test_collect_egl_render_pool_preserves_explicit_pool -q`
  - 结果：2 failed，`_ensure_collect_render_device_pool` 尚不存在。
- GREEN/focused：`conda run -n dreamervla python -m pytest tests/unit_tests/test_mainline_egl_defaults.py tests/unit_tests/test_ray_coldstart_real_config.py tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_runner_injects_top_level_render_backend_into_real_env_cfg tests/unit_tests/test_manual_cotrain_ray_runner.py::test_manual_runner_real_render_backend_overrides_real_env_cfg_backend tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q`
  - 结果：108 passed in 9.46s。
- Lint：`conda run -n dreamervla ruff check dreamervla/runners/cold_start_ray_collect_runner.py tests/unit_tests/test_mainline_egl_defaults.py tests/unit_tests/test_ray_coldstart_real_config.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
  - 结果：All checks passed。
- Static compose/probe：脚本默认 `render_backend=egl`，multi_gpu `ray_online_real_render_backend=None`，collect command 含 `env.cfg.render_backend=egl`，cotrain command 含 `render_backend=egl` 与 `env.cfg.render_backend=egl`，`collect_rollouts_ray` 和 `openvla_onetraj_libero_cotrain_ray` compose 默认均为 `egl`。
- GPU availability：`nvidia-smi --query-gpu=index,name --format=csv,noheader`
  - 结果：8 张 `NVIDIA H100 80GB HBM3` 可见。
- Direct LIBERO EGL probe：`CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 conda run -n dreamervla python -m dreamervla.diagnostics.smoke_libero_online_env --task-id 0 --steps 1 --warmup-steps 0`
  - 结果：reset/step 成功。
- Direct full-record EGL probe：同一 env 直接 `full_record()` before/after step。
  - 结果：`agentview_rgb` 与 `eye_in_hand_rgb` 均返回 `(256, 256, 3)`，无崩溃。
- Real collect EGL smoke：`CUDA_VISIBLE_DEVICES=0 ... python -m dreamervla.train experiment=collect_rollouts_ray ... env.num_workers=1 env.cfg.render_backend=egl ...`
  - 结果：先前零 GPU EGL 拒绝已消失，EnvWorker 与 OFT policy 均启动；但第一步时 `EnvWorker egl child died (rank=0, slot=0)`，父进程收到 `EOFError`，未完成 1 episode。
- Real collect osmesa fallback：同一命令仅改 `env.cfg.render_backend=osmesa`。
  - 结果：1 episode 完成，`COLDSTART COLLECT — done · 1 episodes · succ 0.000`。

## 结论

`BLOCKED`。配置默认、fallback 语义、零 GPU 拒绝和 collect render pool 注入均已落地并有单测覆盖；但 GPU 上真实 Ray collect 的 EGL 子进程仍在 step 阶段死亡，未满足“有 GPU 时 egl 端到端冒烟不崩”判据。

## 下一步建议

下一轮不要进入 R1/R4；先继续 Step 2c blocker。参考 `/mnt/data/spoil/workspace/RLinf` 的 LIBERO `OffScreenRenderEnv`/venv 子进程组织，重点对齐 Ray actor 内 spawn 子进程的 EGL env、robosuite 初始化顺序、以及 child stderr/native crash 捕获。若仍不能稳定，按 SPEC 失败预案为该路径回退 osmesa 并保留默认 EGL 的静态护栏说明。

## 残留风险

- 当前默认已切 EGL，但真实 Ray collect EGL smoke 失败；在 blocker 解决前，真实主线需要显式 `render_backend=osmesa` 才能完成 collect。
- cotrain-real 的真实 EGL smoke 尚未运行；collect 子进程问题解决前不应声称 cotrain-real 通过。
