# RLinf / DreamerVLA EGL 多环境渲染崩溃排查 findings

Date: 2026-06-26

## Scope / 问题

DreamerVLA 的 online cotrain 在 **多环境并行 + egl 渲染** 下经常崩溃。本文对比
DreamerVLA 与上游 RLinf 的 EGL 实现，定位"本质区别"。

对比对象:

- 我们(noray 路径):`dreamervla/envs/online_egl_venv.py`、`dreamervla/runners/online_cotrain_runner.py`
- 我们(ray 路径):`dreamervla/workers/env/env_worker.py`、`dreamervla/runners/online_cotrain_ray_runner.py`
- RLinf:`/mnt/data/spoil/workspace/RLinf/rlinf/envs/libero/`、
  `rlinf/envs/venv/venv.py`、`rlinf/scheduler/hardware/accelerators/nvidia_gpu.py`、
  `rlinf/scheduler/worker/worker_group.py`、`examples/embodiment/config/*.yaml`
- robosuite(vendored):`third_party/robosuite/robosuite/utils/binding_utils.py`、
  `.../renderers/context/egl_context.py`

## 现场:崩溃到底长什么样(已确认)

日志 `logs/ray_egl_g45_20260626_*.log`(今天的 Ray+egl 跑,GPU 4/5):

```
[EnvWorker] egl spawn child died (rank=N); respawn X/5, dropping partial episode   # 所有 rank 反复
...
RuntimeError: EnvWorker egl child died 6 times (rank=0); exceeded egl_max_respawns=5
```

关键证据(`env_worker.py:230`):父进程在 `step()` 里 `self._conn.recv()` 拿到
**`EOFError`** 才发现孩子死了——**没有 Python traceback**。我们的 `_worker` 对任何
Python 异常都会 `send(("error", ...))`;既然父进程只看到管道被关,说明子进程是
**在 native 层被信号打死的(SIGABRT/SIGSEGV)**,发生在 **rollout 中途的 `step`**,
不是 init。这就是 robosuite/mujoco 的 **EGL `read_pixels` 在并发下 abort**
(`env_worker.py:3-6` 注释已认出过这点)。

**重要:** 崩溃时 GPU 4/5 只用了 **~17GB / 81GB**(`nvidia-smi`),**不是 OOM**,还有
~64GB 空闲。所以"显存被占满导致 EGL 崩"这个解释**站不住**。

## 已验证的实现差异(带 file:line)

### 1. EGL 设备 regime 的粒度:per-worker(RLinf) vs per-child round-robin(我们)

- RLinf 在 **worker 进程级别** 只设一次(`nvidia_gpu.py:114`
  `get_accelerator_env_var` → `worker_group.py:270`):
  `CUDA_VISIBLE_DEVICES = <该worker的卡列表>`、`MUJOCO_EGL_DEVICE_ID = visible_accelerators[0]`。
  同一个 `SubprocVectorEnv` 的所有子 env 共享这一个 EGL 设备。横向扩展靠多开 env worker。
- 我们在 **每个子进程** 里设(`online_egl_venv.py:46-60`、`env_worker.py:82-92`):
  round-robin 取一个物理 id,并把 `CUDA_VISIBLE_DEVICES` 收窄成 **单卡**,
  和 `MUJOCO_EGL_DEVICE_ID` 设成同一个值。

### 2. `MUJOCO_EGL_DEVICE_ID` 实测是 EGL 枚举的"位置下标",不是 CUDA id(实测)

robosuite `egl_context.py:create_initialized_egl_device_display` 把
`MUJOCO_EGL_DEVICE_ID` 当作 `eglQueryDevicesEXT()` 列表的下标
(`candidates = all_devices[idx:idx+1]`),并 bounds-check `0 <= idx < len(all_devices)`。

本机实测(dreamervla env):

```
eglQueryDevicesEXT() 返回 10 个 EGL 设备(实际 8 块 GPU)
CVD=0 / CVD=3 / 不设  → 都返回 10(不随 CUDA_VISIBLE_DEVICES 过滤)
```

结论:
- **不会越界崩溃**(idx 0–7 < 10)——我最初怀疑的 bounds-check 崩溃被**证伪**。
- 但 `MUJOCO_EGL_DEVICE_ID=k` **不可靠地落在物理 GPU k 上**(EGL 枚举有 10 项、
  顺序不保证等于 CUDA/NVML)。我们精心做的 "device alignment" 实际并不保证 env
  落到预期的卡,可能多个 EGL context 撞同一块物理卡。

### 3. `MUJOCO_GL` / `PYOPENGL_PLATFORM` 设置时机:OS 级(RLinf) vs split-brain(我们)

- RLinf 在 **启动 shell** OS 级导出(`examples/embodiment/run_embodiment.sh:7-8`
  `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`),早于任何 import,父子一致。
  RLinf 的 env 代码里**没有**任何地方 `os.environ["MUJOCO_GL"]=...`(grep 确认)。
- 我们父进程在 import 时强制 `osmesa`(`online_cotrain_runner.py:39-40`,避免父进程
  robosuite SIGABRT),再在子进程翻回 `egl`(`_EglEnvFn.__call__` /
  `_env_subprocess_main:80`)。父 osmesa、子 egl 的 split-brain,只靠 robosuite import
  被推迟到翻转之后才不炸,较脆。

### 4. respawn / spawn-stagger 是我们独有的"创可贴",RLinf 没有

`env_worker.py:228-233`(`egl_max_respawns=5`)、`:259-269`(按 rank stagger spawn)。
它们的存在本身就是信号:一直在补救 native crash,而没消除根因。

## 纠正:我前两轮说错的两点

1. **"渲染卡 ≠ 算力卡"是错的。** RLinf 的 libero 配置 `libero_10_ppo_dexbotic_pi0.yaml`
   用 `component_placement: actor,env,rollout: all`——env 渲染和训练/推理**同卡**
   (collocated),`total_num_envs: 16`,单机即**一张卡多个 env**。
2. **"一张卡最多一个 env"也是错的。** RLinf 配置里常见 `total_num_envs: 16/64/256`,
   一张卡多个 env。

## 修正后的结构性差异:同卡共享时 RLinf 靠 offload / 分离布局

RLinf 文档(`docs/source-zh/.../examples/embodied/calvin.rst`)给了三种布局:

- `env,rollout,actor: all` —— **完全共享**(同卡)
- `env:0-1 / rollout:2-5 / actor:6-7` —— **完全分离,无干扰,消除了卸载的需要**

关键:`libero_10_ppo_dexbotic_pi0.yaml` 里每组件有 `enable_offload`:

```yaml
# env 段:    enable_offload: False
# rollout 段: enable_offload: True
# actor 段:   enable_offload: True
```

即 **同卡共享时,跑 env/rollout 阶段会把训练/推理模型 offload 出显存**(代码:
`rlinf/hybrid_engines/fsdp/utils.py`、`megatron_model_manager.py`、
`vllm/.../worker.py` 等),所以不会"满载算力 + 还在 EGL 渲染"。要么 offload,要么
干脆把 env 放到独占卡(分离布局,`enable_offload: False`)。

**我们两样都没做:** `online_cotrain_ray_runner.py` 没有模型 offload(只有 1383 行把
一个 metric tensor `.cpu()`,不是权重卸载),OFT 模型全程常驻,EGL 子进程在同卡渲染。
这是和 RLinf 最实质的结构差异。

## 未证实 / 待定(诚实缺口)

崩溃时显存只用了 17/81GB,**不是 OOM**。所以即便 offload/分离是 RLinf 的设计差异,
它也未必是"我们崩"的直接死因。子进程在大量空闲显存下仍 `read_pixels` SIGABRT,更像:

- **多进程并发下 EGL context / driver 层面**的资源或竞态问题;和/或
- **mujoco 3.8.0 + robosuite 1.4.1(vendored)** 这个版本组合本身的不稳。
  - 我们:`mujoco 3.8.0`、`robosuite 1.4.1`(`third_party/robosuite`)。
  - RLinf 的 mujoco/robosuite 版本**没查到**(走预构建 Docker,requirements 未 pin)
    —— 版本差异这条**未证实**。

### Codex 复审补充(2026-06-26)

在空闲 GPU 1/2/3 上跑最小多进程 EGL 复现时,当前 per-child regime
(`CUDA_VISIBLE_DEVICES=<physical>` 且 `MUJOCO_EGL_DEVICE_ID=<same physical>`)没有进入
rollout 中途 native crash,而是在 env init 阶段先暴露出更早的配置语义错误:

```
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer
between 0 and 0 (inclusive), got 1.
```

把 `MUJOCO_EGL_DEVICE_ID` 改成 `0` 后又撞到 vendored robosuite 的导入期 assert:

```
AssertionError: MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id
specified in CUDA_VISIBLE_DEVICES
```

并且在同一 `dreamervla` conda env 下直接查询:

```
CUDA_VISIBLE_DEVICES=1     -> len(eglQueryDevicesEXT()) == 1
CUDA_VISIBLE_DEVICES=1,2,3 -> len(eglQueryDevicesEXT()) == 1
CUDA_VISIBLE_DEVICES unset -> len(eglQueryDevicesEXT()) == 1
```

结论:本机当前运行上下文里,`MUJOCO_EGL_DEVICE_ID` 确实是 EGL 枚举下标,不能当物理
CUDA id 使用;并且 robosuite 的 import-time assert 与 EGL 下标语义存在张力。Codex 没有
继续用同卡算力 stress 追 native abort 字符串,因为在空闲卡上已经能证明现有设备 regime
不可靠。修复因此优先落到:

- 多环境默认 `osmesa`;
- `egl` 多环境必须显式配置与算力 placement 不相交的 `render_devices`,否则启动早失败;
- 子进程启动时记录/校验 EGL 枚举数量,让下标越界以 Python `ValueError` 暴露,不再静默
  演变成 robosuite init 或 native crash。

## 决定性的下一步(尚未做)

现在 Ray 日志只有父进程侧的 `EOFError`,看不到子进程崩前的 native 报错。需要:

1. 用 GPU 跑一个**最小多进程 EGL 渲染复现**,抓子进程崩前的真实 abort 信息
   (`Framebuffer incomplete` / `GL_INVALID_OPERATION` / EGL error / mujoco assert)。
2. 这条信息才能区分"并发 context 上限 vs 版本 bug vs 别的",从而决定修法。

## 候选修法(待死因确认后再定)

- **A(对齐 RLinf,治本):** 给 env 渲染分离布局(独占非算力卡),或在 rollout 阶段
  offload 算力模型;并修正 device 下标语义,避免 EGL context 撞卡。
- **B(治标但稳,已被现状文档证明):** 多环境并行直接用 `osmesa`(CPU 渲染、零 GPU
  争用),牺牲渲染速度换稳定。
- 注意:spawn 隔离、device 对齐、respawn、stagger 已试过 3+ 次都没解决 →
  按 systematic-debugging 应质疑架构,而不是再补第 4 个参数。

## 关键源码定位速查

| 主题 | 我们 | RLinf |
|------|------|-------|
| EGL 设备 regime | `online_egl_venv.py:46-60`、`env_worker.py:82-92`(per-child) | `nvidia_gpu.py:114` + `worker_group.py:270`(per-worker) |
| spawn vec env | `online_egl_venv.py`(`OnlineEglVecEnv`) | `envs/libero/venv.py`(`ReconfigureSubprocEnv`)+ `envs/venv/venv.py` |
| MUJOCO_GL 设置 | `online_cotrain_runner.py:39-40`(父 osmesa)+ 子进程翻 egl | `run_embodiment.sh:7-8`(OS 级) |
| offload / 布局 | 无 offload(`online_cotrain_ray_runner.py`) | `enable_offload` + `component_placement`(`examples/embodiment/config/*.yaml`) |
| robosuite EGL 选卡 | `third_party/robosuite/.../egl_context.py`(位置下标进 `eglQueryDevicesEXT`) | 同 robosuite |
