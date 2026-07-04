# Loop 报告 — eval EGL 崩溃根因坐实 + 全场景修复

## 目标
用户强调:collect/cotrain/eval 三处真实用 EGL 且不崩、底层调用同一 helper;并要求坐实崩因、解决所有场景。

## 坐实的根因(A/B + 显存采样,推翻多个假设)
- **崩因 = mujoco EGL 与重型 torch 策略在同一物理 GPU 上**,NVIDIA `libnvidia-eglcore` 在 `mjr_readPixels` 内 `abort()`(几百帧后)。
- **非显存**:采样显示崩溃时渲染卡 used ~17.6 GB、**free 63 GB**;留 headroom 无用。
- **非编号错位**:`mujoco/egl/__init__.py:37`、robosuite `egl_context.py:38` 用 `eglQueryDevicesEXT()` 全局索引;我们设全局 id 且 robosuite 断言其 ∈ CUDA_VISIBLE_DEVICES → torch/EGL 一致落同一(正确)卡。
- **非 fork/context 继承**:用 spawn,且崩在成功渲染几百帧之后(非 init)。
- 关键 A/B:probe(纯子进程 EGL / +父 torch 小 matmul,同卡)→ 400 步不崩;真实 eval 同卡 → 崩;torch@0+EGL@7 不相交 → 300 步维持。

## 修复(覆盖所有 GPU 场景)
- eval 的 LIBERO env 改为 **spawn 子进程渲染**(`dreamervla/runners/eval_subproc_env.py`,从 RLinf `rlinf/envs/libero/venv.py` 剥离移植:`get_context("spawn")`+`CloudpickleWrapper` env_fn,child 内先 `apply_libero_render_regime` 再建 `OffScreenRenderEnv`)。
- `_eval_render_gpu_pool` 默认渲染卡与 torch `cuda:0` **不相交**(≥2 卡取 `visible[1:]`);两卡都留在 CUDA_VISIBLE_DEVICES 内满足全局 EGL 索引 + robosuite 断言。
- `_eval_render_regime_params`:egl 但渲染卡与 compute[0] 重叠(单卡/显式池重叠)→ **自动降级 osmesa + 警告**。
- 主线 post-step eval 天然 disjoint(`_post_step_eval_egl_device_id` 取 `eval.gpus` 最后一张 vs torch 第一张)。

## 验证(真实 GPU)
- dreamer eval,2 卡 disjoint,max_steps=300 → 写出 metrics,**0 abort**。
- dreamer eval,单卡,egl → 自动降级 osmesa,300 步 → 写出 metrics,**0 abort**。
- 相关单测 32 passed;compose 6 主线 experiment 全绿;ruff 干净。

## 改动文件
- 新增 `dreamervla/runners/eval_subproc_env.py`(RLinf 移植)。
- `dreamervla/runners/embodied_eval_runner.py`(dreamer eval 接子进程 env + disjoint 渲染参数)。
- `dreamervla/runners/pretokenize_vla_runner.py`(`_eval_render_regime_params` + `_eval_render_gpu_pool` disjoint 默认 + osmesa 降级)。
- 提交:`5ac38e5`(子进程+disjoint)、`5785a1c`(osmesa 降级)。74 archive rename 仍 staged 未动。

## 残留(非 EGL,属 R1 base-SR 轨)
- base-VLA eval(`ckpt_kind=vla`,进程内渲染)被 `RynnVLAEncoder` 默认配置 bug(`'NoneType'.get`)挡在渲染前 = SPEC §4 encoder 问题;单卡走 osmesa 降级,≥2 卡进程内 disjoint EGL 因该 bug 未测。
- eval SR=0.0 = 策略质量,非 EGL。

## 下一步
- 修 SPEC §4 eval 默认 encoder(RynnVLA→主线),解锁 base-VLA 全长跑验证。
- (可选)把 base-VLA 进程内渲染也接 EvalSubprocEnv,与 dreamer 路径同构。
- 回到 R1:base-SR vs 5-step cotrain SR 趋势;R4b 逐批归档;Step 5 文档。
