# EGL/render GPU 冒烟验证结果

日期: 2026-06-27

## 本轮结论: Ray/RLinf worker-level EGL 主线

本轮验证的是 Ray 主线,不是旧 no-Ray `OnlineEglVecEnv` 路径。实现按 RLinf 结构运行:

- EnvWorker 由 `cluster.component_placement.env` 绑定 render GPU。
- WorkerGroup runtime env 写入 `CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID`、
  `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。
- 一个 EnvWorker 内通过 `env.envs_per_worker` 承载多个 LIBERO spawn child; child
  继承 worker-level EGL regime,不再各自 round-robin 改卡。
- native child death 直接失败,没有 respawn 掩盖。

卡划分:

- `CUDA_VISIBLE_DEVICES=0,1`
- EnvWorker/render: GPU 0 (`cluster.component_placement.env: 0`)
- Rollout inference + learner/actor: GPU 1 (`cluster.component_placement.rollout,actor: 1`)

EGL 压力命令:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 \
  conda run -n dreamervla env PYTHONPATH=. python -m dreamervla.train \
  experiment=online_cotrain_ray_oft_action_hidden logger=tensorboard \
  render_backend=egl env.num_workers=1 env.envs_per_worker=4 \
  rollout.steps=160 rollout.min_replay_episodes=999999 \
  rollout.min_replay_transitions=999999 training.max_steps=1 \
  training.out_dir=/tmp/dvla_ray_egl_rlinf_stress_20260627_031738 \
  init.warmup_ckpt_path=data/outputs/coldstart_warmup_cotrain/20260625_230217/cotrain/ckpt/ray_async_init.ckpt
```

关键日志:

```text
(EnvWorker pid=3187712) [egl_device] EGL device diagnostics: eglQueryDevicesEXT count=10, MUJOCO_EGL_DEVICE_ID=0 (EGL enumeration index, not CUDA id)
ONLINE COTRAIN (ray) · envs=4
env_steps=87/160 collect=t0:s22,t1:s22,t2:s22,t3:s21
env_steps=160/160 collect=t0:s40,t1:s40,t2:s40,t3:s40
FINAL METRICS: rollout/steps=160 train/learner_updates=0
```

负向 grep:

- 无 `EOFError`
- 无 `egl spawn child died`
- 无 `respawn`
- 无 `child died`
- 无 `SIGABRT` / `abort`
- 无 `Traceback` / `RuntimeError`

结论: Ray EGL worker-level 方案在 1 个 EnvWorker、4 个 EGL LIBERO child、160 env
steps 下稳定推进,没有复现 no-Ray 旧路径的中途 EOF/native crash。

osmesa 默认对照命令:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 \
  conda run -n dreamervla env PYTHONPATH=. python -m dreamervla.train \
  experiment=online_cotrain_ray_oft_action_hidden logger=tensorboard \
  render_backend=osmesa env.num_workers=2 env.envs_per_worker=4 \
  rollout.steps=4 rollout.min_replay_episodes=999999 \
  rollout.min_replay_transitions=999999 training.max_steps=1 \
  training.out_dir=/tmp/dvla_ray_osmesa_default_smoke_20260627_032413 \
  init.warmup_ckpt_path=data/outputs/coldstart_warmup_cotrain/20260625_230217/cotrain/ckpt/ray_async_init.ckpt
```

结论: osmesa 默认路径稳定完成 4/4 env steps;`env.envs_per_worker=4` 被正确忽略为
每个 EnvWorker 单 env,未触发 slot/in-process 冲突。

测试收尾:

```text
conda run -n dreamervla env PYTHONPATH=. pytest -q tests/unit_tests -q  # passed
conda run -n dreamervla env PYTHONPATH=. ruff check .                  # passed
```

备注: Ray 在单测和 smoke 中提示 `/tmp/ray` 所在分区超过 95% full;这是机器磁盘水位
警告,不是 EGL/render 崩溃。

## 机器与卡划分

初始 `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits`:

```text
0, 4, 0
1, 4, 0
2, 4, 0
3, 4, 0
4, 59181, 100
5, 59373, 100
6, 34251, 0
7, 39567, 0
```

已有进程占用:

- GPU 4/5: `/home/user01/miniconda3/envs/dreamervla/bin/python`
- GPU 6/7: `/home/user01/miniconda3/envs/llama-env/bin/python3.12`

本次只使用空闲卡:

- compute: GPU 2 (`CUDA_VISIBLE_DEVICES=2`)
- EGL render: GPU 0 (`online_rollout.render_devices=[0]`)
- osmesa 对照: GPU 2 compute, 默认 `online_rollout.render_backend=osmesa`

## 反向验证: 相交 render/compute 早失败

命令:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 WANDB_MODE=offline \
  DVLA_DATA_ROOT=/mnt/data/spoil/workspace/DreamerVLA/data \
  /home/user01/miniconda3/bin/conda run -n dreamervla python -m dreamervla.train \
  experiment=online_cotrain_oft_action_hidden training.debug=true \
  '+online_rollout.num_envs=4' '+online_rollout.render_backend=egl' \
  '+online_rollout.render_devices=[2]' \
  training.out_dir=/tmp/dvla_egl_overlap_20260627_0207 \
  runner.logger.wandb_mode=offline
```

结果: 通过。完整 train 入口在进入 rollout 前抛出预期错误:

```text
ValueError: online_rollout.render_devices must not overlap compute devices for multi-env egl; render_devices=[2], compute_devices=[2], overlap=[2]. Set disjoint online_rollout.render_devices, or use render_backend=osmesa.
```

备注: `experiment=online_cotrain_oft_action_hidden` 的 `online_rollout.num_envs/render_backend/render_devices`
不在 struct config 中, 命令需要用 `+online_rollout.*` 追加这些键。

## EGL disjoint 多环境正例

直接 `experiment=online_cotrain_oft_action_hidden` 会在 vectorized path 报
`requires an OFT action_hidden extractor`;因此使用计划中 GPU smoke 注释对应的
`experiment=online_cotrain_pipeline_oft_action_hidden_smoke`。默认离线 seed shard
`OpenVLA_Onetraj_LIBERO_libero_goal/.../ray_shard_000.hdf5` 是 96-byte 截断文件,
会导致 empty replay;本次用已有匹配维度的 warmup ckpt 做临时 resume:

```text
source: data/outputs/coldstart_warmup_cotrain/20260625_230217/cotrain/ckpt/
tmp out_dir: /tmp/dvla_egl_pipeline_resume_20260627_0225
```

命令:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 WANDB_MODE=offline \
  DVLA_DATA_ROOT=/mnt/data/spoil/workspace/DreamerVLA/data \
  /home/user01/miniconda3/bin/conda run -n dreamervla python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden_smoke \
  training.resume=true training.checkpoint_format=none \
  online_rollout.render_backend=egl 'online_rollout.render_devices=[0]' \
  online_rollout.debug_total_env_steps=8 online_rollout.debug_min_replay=48 \
  online_rollout.debug_max_train_updates=0 online_rollout.debug_episode_horizon=8 \
  training.out_dir=/tmp/dvla_egl_pipeline_resume_20260627_0225 \
  runner.logger.wandb_mode=offline
```

关键日志:

```text
[online-cotrain] vectorized rollout: 4 envs, render_backend=egl
cotrain [...] 4/8 (50%)
cotrain [...] 8/8 (100%)
```

负向 grep:

- 无 `egl spawn child died`
- 无 `respawn`
- 无 `exceeded egl_max_respawns`
- 无 `EOFError`
- 无 traceback / runtime error

诊断行状态(旧 no-Ray run): smoke 主日志没有出现 `EGL device diagnostics`。单独直接调用
`dreamervla.utils.egl_device.apply_egl_device_regime(0)` 时可以输出:

```text
INFO:dreamervla.utils.egl_device:EGL device diagnostics: eglQueryDevicesEXT count=10, MUJOCO_EGL_DEVICE_ID=0 (EGL enumeration index, not CUDA id)
```

结论: disjoint EGL 在旧 no-Ray 8-step 多环境 smoke 中稳定推进。本轮 Ray 主线已在
上方补齐 worker-level `EGL device diagnostics` 可观测性;旧 no-Ray 主日志是否输出
info 级诊断未重新验证。

## osmesa 默认路径对照

命令:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 WANDB_MODE=offline \
  DVLA_DATA_ROOT=/mnt/data/spoil/workspace/DreamerVLA/data \
  /home/user01/miniconda3/bin/conda run -n dreamervla python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden_smoke \
  training.resume=true training.checkpoint_format=none \
  online_rollout.debug_total_env_steps=8 online_rollout.debug_min_replay=48 \
  online_rollout.debug_max_train_updates=0 online_rollout.debug_episode_horizon=8 \
  training.out_dir=/tmp/dvla_osmesa_pipeline_resume_20260627_0230 \
  runner.logger.wandb_mode=offline
```

关键日志:

```text
[online-cotrain] vectorized rollout: 4 envs, render_backend=osmesa
cotrain [...] 4/8 (50%)
cotrain [...] 8/8 (100%)
```

结论: osmesa 默认路径稳定完成 8/8 env steps,无 EGL/respawn/EOFError 崩溃信号。

## 清理状态

本次 smoke 结束后, GPU 0/1/2/3 均回到空闲低显存状态;未清理或触碰 GPU 4/5/6/7 上的既有进程。
