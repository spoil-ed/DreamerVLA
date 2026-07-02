# OpenVLA-OFT One-Trajectory LIBERO Cotrain

## 简介

推荐入口是 Ray async + EGL。它会自动执行 collection、warmup，然后进入 manual async cotrain：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=egl \
  > logs/cotrain_ray_async_egl.log 2>&1
```

运行前默认已经在仓库根目录，`logs/` 已创建，`DVLA_DATA_ROOT` 指向包含
`checkpoints/` 和 `datasets/` 的数据根；未设置时脚本使用仓库内的 `data/`。

任务入口只保留 shorthand：

```text
task=goal|object|spatial|10
```

拆阶段运行时，`RUN_ROOT` 必须在 warmup 和 cotrain 之间保持一致：

```bash
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<run>"
```

## 完整 Train

这是推荐入口。它会顺序执行 rollout collection、seed replay、world model/classifier
warmup，以及 manual async cotrain。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=egl \
  > logs/cotrain_ray_async_egl.log 2>&1
```

主要产物：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/
${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<run>/collect/
${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<run>/cotrain/
```

## 单独 Collect

只采集 cold-start rollout 数据，写入后续 warmup/cotrain 共同消费的统一目录。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=openvla_onetraj_coldstart_libero \
  logger=tensorboard \
  ++collect.hdf5_reward_dir="${DVLA_DATA_ROOT}/collected_rollouts/libero_goal/reward" \
  ++collect.hidden_dir="${DVLA_DATA_ROOT}/collected_rollouts/libero_goal/hidden" \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/collect_rollouts/libero_goal" \
  > logs/collect_ray.log 2>&1
```

## 单独 Warmup

从已采集的 rollout 数据启动，只写 warmup checkpoint，不进入在线 cotrain。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  skip_collect=true cotrain_phase=warmup render_backend=egl \
  > logs/cotrain_warmup_egl.log 2>&1
```

主要产物：

```text
${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt
```

## 单独 Cotrain

从同一个 `RUN_ROOT` 下的 warmup checkpoint 继续，只运行 online manual async cotrain。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  cotrain_phase=online render_backend=egl \
  > logs/cotrain_ray_async_online_egl.log 2>&1
```

常用日志检查：

```bash
tail -f logs/cotrain_ray_async_egl.log
tail -f logs/cotrain_warmup_egl.log
tail -f logs/cotrain_ray_async_online_egl.log
```

## W&B

cotrain 日志默认写在 run root 下面；离线环境保留 W&B offline run，联网机器再同步。

```text
${RUN_ROOT}/cotrain/log/wandb/all/wandb/offline-run-*
```

共享盘同步入口：

```bash
bash scripts/run_wandb_relay_sync.sh --dry-run --once
bash scripts/run_wandb_relay_sync.sh --once
bash scripts/run_wandb_relay_sync.sh
```

运行前只需要编辑 `scripts/run_wandb_relay_sync.sh` 顶部的 `WANDB_DIR`、
`WANDB_PROJECT` 和 `WANDB_ENTITY`。
