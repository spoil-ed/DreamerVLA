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
