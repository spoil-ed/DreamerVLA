# OpenVLA-OFT One-Trajectory LIBERO Cotrain

## 入口

当前主线是 OpenVLA-OFT one-trajectory cold start：

```text
collect -> warmup world model/classifier -> manual Ray async cotrain -> eval
```

推荐只通过 launcher 进入，不手写底层 `python -m dreamervla.train`
collection 命令。launcher 会补齐 reward/hidden 输出目录、OpenVLA-OFT
latent spec、Ray worker 数、warmup checkpoint 和 async online init 路径。

先在仓库根目录设置数据根并创建日志目录：

```bash
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
mkdir -p logs
```

`DVLA_DATA_ROOT` 必须包含当前任务需要的资产：

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1
${DVLA_DATA_ROOT}/datasets/libero/libero_goal
```

任务 shorthand：

```text
task=goal|object|spatial|10
```

Ray async wrapper 会从 `CUDA_VISIBLE_DEVICES` 推导 `ngpu`；显式传
`ngpu=<N>` 时以显式值为准。默认渲染后端是 `osmesa`，主线命令也固定写出
`render_backend=osmesa`，避免误用 EGL。

## 先 Dry Run

运行前先 dry-run，确认实际生成的 collect、warmup、online 命令和路径：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal render_backend=osmesa dry_run=true
```

dry-run 只打印计划，不检查资产、不启动训练。
Ray collection 计划中应能看到底层 route `experiment=collect_rollouts_ray`；
cotrain online 计划中应能看到 `experiment=openvla_onetraj_libero_cotrain_ray`。

## 完整训练

这是推荐入口，会自动执行 collection、warmup、consolidate warmup checkpoint，
然后进入 manual Ray async cotrain：

```bash
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/openvla_goal_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa \
  > logs/cotrain_ray_async.log 2>&1
```

主要产物：

```text
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/reward
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/hidden
${DVLA_DATA_ROOT}/collected_rollouts/<suite>/collection_manifest.json
${RUN_ROOT}/collect/
${RUN_ROOT}/cotrain/
```

collection 数据写入稳定的 `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/`
目录；`RUN_ROOT` 只隔离本次 collect/cotrain 的运行产物。

## 拆阶段运行

拆阶段时必须复用同一个 `RUN_ROOT`。

先运行 collection + warmup，只写 warmup checkpoint，不进入 online cotrain：

```bash
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/openvla_goal_manual"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  cotrain_phase=warmup render_backend=osmesa \
  > logs/cotrain_warmup.log 2>&1
```

如果 `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/` 已经采集完整，并且只想重跑
warmup，追加 `skip_collect=true`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  skip_collect=true cotrain_phase=warmup render_backend=osmesa \
  > logs/cotrain_warmup.log 2>&1
```

warmup 产物：

```text
${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt
${RUN_ROOT}/cotrain/ckpt/ray_async_init.ckpt
```

然后从同一个 `RUN_ROOT` 进入 online manual async cotrain：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" \
  cotrain_phase=online render_backend=osmesa \
  > logs/cotrain_ray_async_online.log 2>&1
```

`cotrain_phase=online` 会跳过 collection，并要求 `${RUN_ROOT}/cotrain/ckpt/`
下已有 warmup checkpoint；缺失时 asset check 会提前失败。

## 常用覆盖

缩小到短 smoke/debug。`debug=true` 会同时缩小 cold-start collection、
warmup 和 async online cotrain 预算；用户显式传入的 `collect.*`、
`warmup.*` 或 `manual_cotrain.*` 覆盖仍然优先生效：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa \
  debug=true
```

指定采集目标总 episode 数。launcher 会统计已有完整 reward/hidden shard，
不足时只补采缺口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa \
  collect_target_episodes=500 collect_num_tasks=10
```

no-Ray 同步路径用于排查 Ray 问题，不是推荐 async 主路：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal run_root="${RUN_ROOT}" render_backend=osmesa
```

## 日志和检查

```bash
tail -f logs/cotrain_ray_async.log
tail -f logs/cotrain_warmup.log
tail -f logs/cotrain_ray_async_online.log
```

检查已采集数据是否完整：

```bash
python -c "from pathlib import Path; import os; from dreamervla.dataset.collection_manifest import summarize_collection, format_collection_report; root = Path(os.environ.get('DVLA_DATA_ROOT', 'data')) / 'collected_rollouts/libero_goal'; print(format_collection_report(summarize_collection(root / 'reward', root / 'hidden', target_total=500, num_tasks=10), root=root))"
```

## W&B

cotrain 日志默认写在 run root 下。离线环境保留 W&B offline run，联网机器再同步：

```text
${RUN_ROOT}/cotrain/log/wandb/all/wandb/offline-run-*
```

共享盘同步入口：

```bash
bash scripts/run_wandb_relay_sync.sh --dry-run --once
bash scripts/run_wandb_relay_sync.sh --once
bash scripts/run_wandb_relay_sync.sh
```

运行前编辑 `scripts/run_wandb_relay_sync.sh` 顶部的 `WANDB_DIR`、
`WANDB_PROJECT` 和 `WANDB_ENTITY`。
