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
cd /mnt/data/spoil/workspace/DreamerVLA
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

## W&B Offline Relay Sync (Air-Gapped GPU Training)

这一节解决一个常见部署问题：**GPU 训练机不能联网，CPU 机器可以联网**。GPU 机只能把
W&B 日志写成本地 offline run，CPU "relay" 机器周期性地把这些 offline run 拉过来再上传到
W&B cloud。

它是一个独立的辅助工具，不改训练流程：

- `scripts/wandb_relay_sync.py` — relay 主程序（周期性 rsync + `wandb sync`）。
- `scripts/run_wandb_relay_sync.sh` — 启动模板，先改占位符再运行。

注意这是 **near-real-time（近实时）**，不是严格实时：CPU 机每隔一段时间同步一轮。每轮独立、
可恢复、幂等；某一轮失败不会影响 GPU 机的训练，下一轮继续重试。

### GPU 机：写本地 offline run（训练命令不用改）

训练命令本身不变，只要保证 W&B 写到本地、用 offline 模式。两种方式任选其一：

仓库原生方式（推荐，走 Hydra logger 组）：

```bash
# 在已有训练命令后追加 logger 覆盖即可
bash scripts/e2e_manual_cotrain_async.sh \
  resume=false gpus=1,2,3,4,5 \
  logger=tensorboard_wandb runner.logger.wandb_mode=offline \
  > logs/cotrain_manual_async_offline.log 2>&1
```

或者用环境变量（训练前 export，命令不变）：

```bash
export WANDB_MODE=offline
export WANDB_DIR="${DVLA_DATA_ROOT}/outputs"     # 或你明确指定的日志目录
export WANDB_PROJECT="dreamervla"
export WANDB_ENTITY="your-wandb-entity"
```

offline run 会写到 cotrain run 的日志目录下，实际落点形如：

```text
${DVLA_DATA_ROOT}/outputs/<run>/cotrain/log/wandb/.../wandb/offline-run-*
```

其中**直接包含 `offline-run-*` 的那个 `wandb/` 目录**，就是后面 relay 要填的
`--remote-wandb-dir`。GPU 机这边到此为止，relay 只读取、绝不修改它。

### CPU relay 机：登录 W&B 并准备 SSH 访问

```bash
pip install wandb          # 确保有 `wandb` 命令
wandb login                # 或：export WANDB_API_KEY=...（只在本机，别写进 repo）
```

确认能从 relay 机用 SSH/rsync 访问 GPU 机（建议配置免密钥登录）：

```bash
ssh -p 22 your-remote-user@gpu-host.example.com 'ls /path/to/.../cotrain/log/wandb'
```

如果**不能直接 SSH**，可以用 NFS / 共享文件系统 / 其他文件同步方式，把 GPU 机的 `wandb/`
目录暴露到 relay 机本地，再把 `--remote-host` 指向可达地址、`--remote-wandb-dir` 指向能读到
offline run 的路径即可。

### dry-run 自检（不连接、不上传）

先编辑 `scripts/run_wandb_relay_sync.sh` 顶部的占位符（host、user、路径、project、entity），
然后做一次 dry-run，确认拼出来的 `rsync` 和 `wandb sync` 命令正确：

```bash
bash scripts/run_wandb_relay_sync.sh --dry-run --once
```

dry-run 只打印将要执行的命令，不会连接 GPU 机，也不会上传 W&B。

### 单次真实同步（第一次验证）

```bash
bash scripts/run_wandb_relay_sync.sh --once
```

它会拉取一次远端 `wandb/` 到本地 mirror，然后 `wandb sync` 上传。退出码 `0` 表示这一轮
成功，非 `0` 表示这一轮失败（看日志原因）。

### 长期运行

```bash
bash scripts/run_wandb_relay_sync.sh
```

默认每 60 秒一轮。建议放进 `tmux` 或 `nohup`：

```bash
nohup bash scripts/run_wandb_relay_sync.sh > wandb_relay.out 2>&1 &
```

Ctrl-C 或 `kill <pid>`（SIGTERM）会在当前一轮后干净退出。本地 mirror 目录默认不清理，
也绝不会删除 GPU 机上的任何文件。

### 常见问题（FAQ）

- **`wandb: command not found`**：relay 机没装 wandb。`pip install wandb`，或用
  `--wandb-bin /full/path/to/wandb` 指定绝对路径。
- **上传失败、提示未登录 / API key 错误**：在 relay 机执行 `wandb login`（或正确设置
  `WANDB_API_KEY`）。API key 只放 relay 机，不要写进 repo、脚本或配置。
- **云端找不到 run**：通常是 `--wandb-project` / `--wandb-entity` 配错。确认它们和你
  W&B 账号下的 project/entity 一致；relay 命令会显式带上 `-p`/`-e`。
- **rsync permission denied**：relay 机没有读 GPU 机目录的权限。确认 SSH 用户、SSH key、
  以及该用户对 `--remote-wandb-dir` 有读权限。relay 只读，不需要写权限。
- **正在写入的 run 第一轮上传不完整**：训练还在写的 offline run，第一轮可能只传到当时的
  进度，这是正常的；后续轮次会继续同步，训练结束后的某一轮会把它补全。
- **不小心启动了多个 relay**：本地 mirror 下的 lock file（默认
  `<local-mirror-dir>/.wandb_relay.lock`）会阻止并发；第二个进程会报错退出。
- **它是 near-real-time，不是实时**：每轮之间有 `--interval` 间隔；需要更快就调小 interval，
  但这只是缩短延迟，不会变成严格实时流式上传。
