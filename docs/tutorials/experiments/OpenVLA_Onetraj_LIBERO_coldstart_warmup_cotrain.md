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
