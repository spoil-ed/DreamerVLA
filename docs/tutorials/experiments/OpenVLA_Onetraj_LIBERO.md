# OpenVLA-OFT One-Trajectory LIBERO Cotrain

## Overview

这份文档是 OpenVLA-OFT one-trajectory LIBERO 主线的最小操作说明。目标是跑通并长期运行这条流程：

```text
collect rollouts -> seed replay -> warmup world model/classifier -> online manual cotrain
```

本文只保留实际使用时需要的顺序、命令和产物位置。更完整的背景说明可以看：

- [EXPLAINED.md](EXPLAINED.md)
- [../../PARAMETERS.md](../../PARAMETERS.md)
- [../../install.md](../../install.md)
- [../../../scripts/README.md](../../../scripts/README.md)

任务 shorthand：

```text
task=goal|object|spatial|10
```

cotrain 之前的离线 world model warmup 走 discrete-token WM 路由
`experiment=oft_discrete_token_world_model_chunk`（DreamerVLA 别名
`experiment=dreamervla_oft_discrete_token_wm_lumos`）。

## Setup

先进入仓库，设置数据目录，安装环境：

```bash
cd DreamerVLA
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"

bash scripts/install_env.sh
conda activate dreamervla
```

下载权重、数据和必要资产：

```bash
bash scripts/download_assets.sh
```

安装后做一次环境检查：

```bash
bash scripts/install/60_verify.sh
```

如果只想下载某一类资产，参考 [../../install.md](../../install.md) 里的 `download_assets.sh only=[...]` 示例。

## Training

推荐入口是 Ray async + EGL。它会自动执行 collection、warmup，然后进入 manual async cotrain：

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu \
  cotrain_engine=async render_backend=egl \
  > logs/cotrain_ray_async_egl.log 2>&1
```

如果要避开物理 GPU0，并且直接使用当前 manual async 主线脚本：

```bash
bash scripts/e2e_manual_cotrain_async.sh \
  resume=false \
  gpus=1,2,3,4,5 \
  > logs/cotrain_manual_async_fresh_5gpu.log 2>&1
```

`gpus=1,2,3,4,5` 会在进程内重新编号，所以程序里的 visible GPU0 对应物理 GPU1。

### 从已有 warmup 权重直接进入 cotrain

如果 collection 和 offline warmup 已经跑完，且 `${RUN_ROOT}/cotrain/ckpt/` 下已经有：

```text
wm_warmup.ckpt
classifier_warmup.ckpt
```

推荐仍然走 coldstart launcher，只把 phase 切到 online。它会跳过 collection/warmup，
必要时把两个 warmup checkpoint 合并成 manual Ray runner 需要的 `ray_async_init.ckpt`：

```bash
mkdir -p logs
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<run>"

DVLA_COTRAIN_HANDSHAKE_TRACE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu \
  cotrain_engine=async cotrain_phase=online \
  run_root="${RUN_ROOT}" render_backend=egl \
  manual_cotrain.envs_per_worker=2 \
  manual_cotrain.global_steps=25 \
  > logs/cotrain_ray_async_online_from_warmup.log 2>&1
```

如果已经有 `${RUN_ROOT}/cotrain/ckpt/ray_async_init.ckpt`（合并好的 warmup
world_model + classifier），直接启动 manual cotrain runner，跳过 collection/warmup：

```bash
mkdir -p logs
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<run>"

DVLA_COTRAIN_HANDSHAKE_TRACE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_NVLS_ENABLE=0 \
python -m dreamervla.train \
  experiment=openvla_onetraj_libero_cotrain_ray \
  task=openvla_onetraj_coldstart_libero \
  init.warmup_ckpt_path="${RUN_ROOT}/cotrain/ckpt/ray_async_init.ckpt" \
  training.out_dir="${RUN_ROOT}/cotrain" \
  manual_cotrain.ngpu=6 +cluster.num_gpus=6 \
  > logs/cotrain_manual_async_from_warmup.log 2>&1
```

- `init.warmup_ckpt_path`：LearnerWorker 载入 world_model + classifier；OFT policy 由
  `task.openvla_oft.ckpt_path` 自动加载。
- RL 超参（`group_size=8` / `rollout_epoch=16` / `max_steps_per_rollout_epoch=256` /
  `env.train.total_num_envs=64` / `global_steps=1000` / gamma / gae_lambda / clip 等）已按
  RLinf 对齐写进配置默认，命令行无需重复；要短跑临时加 `manual_cotrain.global_steps=<N>`。
- `manual_cotrain.ngpu=N`（0–6）= 本机 GPU 数：GPU0=real_env，GPU1..N-1=wm_env，配 N 个
  rollout worker。默认 osmesa，要 EGL 加 `render_backend=egl`。
- `DVLA_COTRAIN_HANDSHAKE_TRACE=1` 打印 env/rollout 握手与 `[build_groups]` 各 worker init
  边界，用于定位卡点。

排障或对比时可以使用以下入口：

```bash
# no-Ray + osmesa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=osmesa \
  > logs/cotrain_noray_osmesa.log 2>&1

# no-Ray + egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=egl \
  > logs/cotrain_noray_egl.log 2>&1

# Ray + osmesa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=osmesa \
  > logs/cotrain_ray_osmesa.log 2>&1

# Ray + egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=egl \
  > logs/cotrain_ray_egl.log 2>&1
```

## Standalone Data Collection（单卡 EGL，多卡并行）

只想**单独采集 rollout 数据**（不跑完整 cotrain，例如补 seed replay、把空闲卡用满）时，用下面这套已验证的配方：**每张空闲卡跑一个独立的单卡 `collect_rollouts_ray` 作业**。这样多卡只是“多份单卡作业”，互不干扰，单卡也能稳定写出。

核心原则：

- **用 EGL 在 GPU 上渲染**（`++env.render_backend=egl`），不要用 osmesa——osmesa 是 CPU 软渲染，GPU 利用率自然很低。
- **每个作业按物理卡号 pin `CUDA_VISIBLE_DEVICES`**，进程内就是 GPU0，OFT 推理 worker 和 EGL 渲染都落在这一张卡上，不会串到别的卡。
- **不要设 `collect.egl_device_pool`、也不要手动 export `MUJOCO_GL`**——让 `render_backend=egl` 这个 config 来驱动 EnvWorker 的 EGL 子进程路径即可。
- `env.num_workers=6`（6 个并行 EGL 环境）是实测稳定的甜点；要更高吞吐优先**多卡并行**，而不是单卡堆 env 数。

单卡启动（存成 `collect_card.sh`，参数：物理卡号、task_ids）：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /path/to/DreamerVLA
export DVLA_DATA_ROOT=$PWD/data
PY=/path/to/miniconda3/envs/dreamervla/bin/python   # 必须用 dreamervla 环境

GPU="$1"; TASKS="$2"; EPISODES="${3:-30}"; NW="${4:-6}"
ROOT=$DVLA_DATA_ROOT/outputs/collect_egl_g${GPU}; G0=$ROOT/coldstart_g0
mkdir -p "$ROOT/logs"

env CUDA_VISIBLE_DEVICES="$GPU" NCCL_NVLS_ENABLE=0 "$PY" -m dreamervla.train \
  experiment=collect_rollouts_ray task=openvla_onetraj_coldstart_libero logger=tensorboard \
  "collect.task_ids=[${TASKS}]" collect.episodes_per_task="$EPISODES" collect.episode_horizon=64 \
  ++env.render_backend=egl env.num_workers="$NW" collect.memory_fraction=0.85 \
  "task.openvla_oft.hdf5_reward_dir=$G0/reward" "task.openvla_oft.input_token_hidden_dir=$G0/hidden" \
  "++collect.hdf5_reward_dir=$G0/reward" "++collect.hidden_dir=$G0/hidden" \
  '++collect.oft_latent_spec.expected_action_head_type=${task.openvla_oft.input_tokens.expected_action_head_type}' \
  '++collect.oft_latent_spec.expected_obs_hidden_source=${task.openvla_oft.input_tokens.expected_obs_hidden_source}' \
  '++collect.oft_latent_spec.expected_prompt_style=${task.openvla_oft.input_tokens.expected_prompt_style}' \
  '++collect.oft_latent_spec.expected_history=${task.openvla_oft.input_tokens.expected_history}' \
  '++collect.oft_latent_spec.expected_include_state=${task.openvla_oft.input_tokens.expected_include_state}' \
  '++collect.oft_latent_spec.expected_rotate_images_180=${task.openvla_oft.input_tokens.expected_rotate_images_180}' \
  '++collect.oft_latent_spec.token_dim=${task.openvla_oft.input_tokens.token_dim}' \
  '++collect.oft_latent_spec.token_count=${task.openvla_oft.input_tokens.token_count}' \
  '++collect.oft_latent_spec.wm_obs_dim=${task.openvla_oft.input_tokens.wm_obs_dim}' \
  '++collect.oft_latent_spec.chunk_size=${task.openvla_oft.input_tokens.chunk_size}' \
  "training.out_dir=$ROOT/collect"
```

> `++collect.oft_latent_spec.*` 这一整段和仓库官方采集脚本
> `configs/scripts/coldstart_warmup_cotrain.yaml`（129–156 行）采集步用的是同一套——
> 它让 collect 侧的 latent 契约对齐 task 配置。`${task.openvla_oft.input_tokens.*}`
> 是 Hydra 插值，**必须用单引号**包住，否则 bash 会先去做变量替换而报 `bad substitution`。

三张空闲卡并行（GPU1/2/3，各占一张卡、各写各的目录、各自后台）：

```bash
nohup bash collect_card.sh 1 "0,1,2,3" 30 6 > logs/collect_g1.log 2>&1 &
nohup bash collect_card.sh 2 "4,5,6"   30 6 > logs/collect_g2.log 2>&1 &
nohup bash collect_card.sh 3 "7,8,9"   30 6 > logs/collect_g3.log 2>&1 &
```

实测（3 卡 × 6 env/卡，`openvla_onetraj_coldstart_libero`）：

- **显存** ≈ 18 GB/卡（OFT 模型本体 ~14 GB + EGL/环境缓冲）。
- **利用率** 峰值 52–72%，呈**突发型**——OFT 是 chunk 推理（每 8 个 env step 才一次 policy 前向），推理与渲染交替，所以不是持续 100%，这是正常现象，不是 bug。
- **产出** 每卡稳定写出 `coldstart_g0/hidden`（12–16 GB）+ `coldstart_g0/reward`（2.2–2.9 GB），3 卡合计约 47 GB 真实数据；shard 体积持续增长即代表在稳定写入。

判断是否健康：`tail -f logs/collect_g1.log` 看到 `collect N · …ep/s` 的 N 在涨，且
`du -h .../coldstart_g0/hidden/ray_shard_000.hdf5` 体积在涨，就是稳定采集。

### 采集踩坑速查（问题 → 解决）

| 现象 | 根因 | 解决 |
| --- | --- | --- |
| GPU 利用率长期个位数 | 用了 osmesa（CPU 软渲染） | `++env.render_backend=egl` |
| env 初始化卡死/极慢 | 只 export 了 `MUJOCO_GL=egl`，没开 `render_backend=egl` config，EnvWorker 没走 EGL 子进程 spawn | 用 config 驱动 EGL，**不要**手动 export `MUJOCO_GL` |
| 静默写出几 KB 空 shard、0 episode | 设了 `collect.egl_device_pool=[<物理卡号>]`，EGL 索引被当成 CUDA 物理 id，设备错位 | 删掉 `egl_device_pool`，改成每卡 pin `CUDA_VISIBLE_DEVICES` |
| 推理 worker 跑到繁忙的 GPU0 上 | 没 pin `CUDA_VISIBLE_DEVICES`，inference 默认落 GPU0 | 每个作业 pin 一张物理卡 |
| 0 episode 失败 | 单卡 env 数堆太高（如 12/卡），EGL context 起不全 | 单卡用 6 env；要更高吞吐靠**多卡并行**，不要单卡堆 env |
| 看着像挂了就被 kill | EGL 首个 env 要建子进程 EGL context，约 1–5 分钟 | 等到看到 `COLDSTART COLLECT` 横幅 / 首个 episode 写出再判断 |

## Validation

启动后先看日志：

```bash
tail -f logs/cotrain_ray_async_egl.log
tail -f logs/cotrain_manual_async_fresh_5gpu.log
```

再看 GPU 是否被持续使用：

```bash
nvidia-smi
```

一次健康的训练应能看到这些现象：

```text
Replay 写入：replay_buffer/transitions 持续增长
Rollout 推理：rollout/generated 非零
Actor 更新：actor/ppo_updates = 1，actor/policy_grad_norm 非零
Learner 更新：wm/loss 和 wm/grad_norm 非零
WMEnv 推理：env/wm_env/batch_size_avg 稳定接近 8
同步推进：policy / rollout / wm / classifier version 持续增长
Checkpoint：checkpoints/manual_cotrain_step_<N>/manual_cotrain.ckpt 写出
```

如果需要看曲线：

```bash
tensorboard --logdir "${DVLA_DATA_ROOT}/outputs" --host 0.0.0.0 --port 6006
```

## Resume

manual cotrain checkpoint 目录通常是：

```text
${DVLA_DATA_ROOT}/outputs/<run>/cotrain/checkpoints/manual_cotrain_step_<N>/
├── manual_cotrain.ckpt
└── manual_cotrain_manifest.json
```

继续训练时只需要传 `resume`、`ckpt` 和 `gpus`：

```bash
bash scripts/e2e_manual_cotrain_async.sh \
  resume=true \
  ckpt="${DVLA_DATA_ROOT}/outputs/<run>/cotrain/checkpoints/manual_cotrain_step_<N>/manual_cotrain.ckpt" \
  gpus=1,2,3,4,5 \
  > logs/cotrain_manual_async_resume.log 2>&1
```

脚本会读取 `manual_cotrain_manifest.json`，识别 checkpoint 的 global step，然后继续向后训练。

## Outputs

常用输出位置：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/
${DVLA_DATA_ROOT}/outputs/<run>/collect/
${DVLA_DATA_ROOT}/outputs/<run>/cotrain/
logs/*.log
```

cotrain run 里最常用的文件：

```text
resolved_config.yaml
run_manifest.json
log/tensorboard/
log/wandb/
checkpoints/manual_cotrain_step_<N>/manual_cotrain.ckpt
checkpoints/manual_cotrain_step_<N>/manual_cotrain_manifest.json
```

其中 `manual_cotrain_manifest.json` 是 resume 的索引文件，记录 `global_step`、actor/rollout/world-model/classifier 版本和 replay 信息。

## W&B Offline Sync (Shared Disk, Air-Gapped GPU)

这一节解决一个常见部署问题：**GPU 训练机不能联网，但和一台能联网的机器共享同一块硬盘**
（NFS / 共享挂载）。GPU 机只能把 W&B 日志写成本地 offline run；在线机器**直接读取共享盘上
的这些 offline run**，周期性地用 `wandb sync` 上传到 W&B cloud。因为是共享盘，**不需要 SSH、
不需要 rsync、也不需要任何拷贝**——在线机直接对共享盘上的目录跑 `wandb sync` 就行。

它是一个独立的辅助工具，不改训练流程：

- `dreamervla.diagnostics.wandb_relay_sync` — 周期性对共享盘上的 `wandb/` 目录跑
  `wandb sync --sync-all`（带 lock、日志、dry-run、错误隔离）。
- `scripts/run_wandb_relay_sync.sh` — 启动模板，先改占位符再运行。

`wandb sync --sync-all` 是**幂等**的：某个 offline run 成功上传后，wandb 会在它旁边写一个
`.synced` 标记，下一轮就跳过它。所以这个循环只会上传**尚未同步过的 run**，不会重复上传。
某一轮失败也不会影响 GPU 机的训练，下一轮继续重试。

### GPU 机：把 offline run 写到共享盘（训练命令不用改）

训练命令本身不变，只要保证两点：W&B 用 **offline 模式**，且 run 的输出目录落在**共享盘**上
（这样在线机才看得到）。让 `DVLA_DATA_ROOT` 指向共享盘即可。

仓库原生方式（推荐，走 Hydra logger 组）：

```bash
export DVLA_DATA_ROOT=/shared/DreamerVLA/data          # 共享盘上的输出根
bash scripts/e2e_manual_cotrain_async.sh \
  resume=false gpus=1,2,3,4,5 \
  logger=tensorboard_wandb runner.logger.wandb_mode=offline \
  > logs/cotrain_manual_async_offline.log 2>&1
```

或者用环境变量（训练前 export，命令不变）：

```bash
export WANDB_MODE=offline
export WANDB_DIR=/shared/DreamerVLA/data/outputs       # 共享盘上的日志目录
export WANDB_PROJECT="dreamervla"
export WANDB_ENTITY="your-wandb-entity"
```

offline run 会写到 cotrain run 的日志目录下，实际落点形如：

```text
/shared/DreamerVLA/data/outputs/<run>/cotrain/log/wandb/all/wandb/offline-run-*
```

其中**直接包含 `offline-run-*` 的那个 `wandb/` 目录**，就是后面要填的 `--wandb-dir`。

### 在线机：登录 W&B

```bash
pip install wandb          # 确保有 `wandb` 命令
wandb login                # 或：export WANDB_API_KEY=...（只在本机，别写进 repo）
```

确认这台机器能在共享盘上读到 offline run（`--wandb-dir` 要填**在线机这边能读到 `offline-run-*`
的那个 `wandb/` 目录**，挂载路径若和 GPU 机不同以在线机为准）：

```bash
ls /shared/DreamerVLA/data/outputs/<run>/cotrain/log/wandb/all/wandb
```

### dry-run 自检（不上传）

先编辑 `scripts/run_wandb_relay_sync.sh` 顶部的占位符（`WANDB_DIR`、project、entity），
然后做一次 dry-run，确认拼出来的 `wandb sync` 命令正确：

```bash
bash scripts/run_wandb_relay_sync.sh --dry-run --once
```

dry-run 只打印将要执行的命令，不会上传 W&B。

### 单次真实同步（第一次验证）

```bash
bash scripts/run_wandb_relay_sync.sh --once
```

它会对共享盘上的 `wandb/` 跑一次 `wandb sync`，把尚未同步的 offline run 上传。退出码 `0`
表示这一轮成功，非 `0` 表示这一轮失败（看日志原因）。

### 长期运行

```bash
bash scripts/run_wandb_relay_sync.sh
```

默认每 60 秒一轮。建议放进 `tmux` 或 `nohup`：

```bash
nohup bash scripts/run_wandb_relay_sync.sh > wandb_relay.out 2>&1 &
```

Ctrl-C 或 `kill <pid>`（SIGTERM）会在当前一轮后干净退出。它绝不会删除或改动 GPU 机写的
run 数据；只有 wandb 自己会在 run 目录旁写 `.synced` 标记和 `debug-cli.log`（见 FAQ）。

### 常见问题（FAQ）

- **`wandb: command not found`**：在线机没装 wandb。`pip install wandb`，或用
  `--wandb-bin /full/path/to/wandb` 指定绝对路径。
- **上传失败、提示未登录 / API key 错误**：在在线机执行 `wandb login`（或正确设置
  `WANDB_API_KEY`）。API key 只放在线机，不要写进 repo、脚本或配置。
- **云端找不到 run**：通常是 `--wandb-project` / `--wandb-entity` 配错。确认它们和你
  W&B 账号下的 project/entity 一致；命令会显式带上 `-p`/`-e`。
- **`--wandb-dir` 找不到 / 读不到**：确认共享盘已挂载、路径指向**直接包含 `offline-run-*`
  的那个 `wandb/` 目录**，且当前用户对它有读权限。
- **会不会动到 GPU 的 run 数据**：本工具自身不写 run；但 `wandb sync` 成功后会在 run 目录旁
  写一个很小的 `.synced` 标记（用于幂等跳过）和一个 `debug-cli.log`。这两者都不碰 run 的
  `.wandb` 数据本身。如果完全不想在共享盘上落任何额外文件，可以先把 `wandb/` 拷到本机再 sync。
- **每个 run 只会上传一次（重要）**：`--sync-all` 上传成功后即标记 `.synced` 并在后续轮次
  跳过它。所以**已经结束的 run 会被完整上传**；而一个**还在写的长 run**，如果在写到一半时被
  同步，只会传到当时的进度、之后不再补传。若想让一个长 run 在训练过程中持续增量上传，请改用
  `wandb sync --append`（本工具默认用 `--sync-all`，更适合按 run 粒度的周期上传）。
- **不小心启动了多个进程**：lock file（默认 `<wandb-dir>/.wandb_relay.lock`）会阻止并发；
  第二个进程会报错退出。可用 `--lock-file` 指定到本机本地路径。
- **它是周期上传，不是实时流**：每轮之间有 `--interval` 间隔；调小 interval 只是缩短延迟，
  不会变成严格实时上传。
