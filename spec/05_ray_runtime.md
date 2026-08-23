# Ray Runtime

Ray/manual cotrain 是当前主线 cotrain 后端。目标不是新增训练语义，而是把 env、rollout、actor、
learner 分成可并行调度的 runtime group。

## Groups

| Group | Worker | 职责 |
| --- | --- | --- |
| `LearnerGroup` | `LearnerWorker` | 更新 world model 和 classifier/reward model。 |
| `ActorGroup` | `EmbodiedFSDPActor` | 训练 VLA policy；FSDP、PPO、optimizer 在这里。 |
| `RolloutGroup` | `MultiStepRolloutWorker` / inference worker | no-grad policy inference，生成 action chunk 和 forward inputs。 |
| `EnvGroup` | `RealEnvWorker` / `WMEnvWorker` | 真实环境或 WMEnv step，组装 trajectory。 |
| `ReplayGroup` | `ReplayWorker` | 可选临时数据服务，用于 replay、warmup 和 bootstrap；不参与 cotrain resume。 |

## Placement

默认单机多 GPU 思路：

```text
GPU0:
  RealEnvWorker + RolloutWorker + LearnerWorker

GPU1..GPU(N-1):
  WMEnvWorker + RolloutWorker + ActorGroup rank
```

`N=0` 用 CPU startup 路径；`N=1` 时所有角色落在 GPU0。具体资源来自 Hydra/Ray placement 配置。

## Sync

- ActorGroup 内部同步由 FSDP/NCCL 管理。
- ActorGroup -> RolloutGroup：同步 VLA policy patch/version。
- LearnerGroup -> WMEnvWorker：同步 world model 和 classifier/reward model。

同步必须记录方向、版本、触发时机和结果。不要让 env step、rollout inference 或 replay sample 隐式更新权重。

## Entrypoints

- `experiment=openvla_onetraj_libero_cotrain`
- `_target_=dreamervla.runners.CotrainRunner`
- shell 入口：`scripts/experiments/cotrain/train.sh`
