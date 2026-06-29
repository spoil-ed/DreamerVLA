# Complete Loop

主线是 OpenVLA-OFT one-trajectory cold start：

```text
collect -> warmup -> online cotrain -> eval
```

## 1. Collect

collection 在真实 LIBERO 中运行当前 VLA/OFT policy，写出 reward shard、hidden sidecar 和
`collection_manifest.json`。

当前入口：

- no-Ray：`experiment=collect_rollouts_onetraj` -> `CollectRolloutsRunner`
- Ray：`experiment=collect_rollouts_ray` -> `ColdStartRayCollectRunner`

collection 只负责产生初始真实数据，不负责训练 world model 或 actor。

## 2. Warmup

cotrain 前先用 collected reward + hidden HDF5 shard 训练：

- world model
- classifier/reward model

sync pipeline 使用 `OnlineCotrainPipelineRunner`。async pipeline 先跑 sync warmup，再把 warmup checkpoint
合并成 manual Ray cotrain 的 init checkpoint。

## 3. Online Cotrain

online cotrain 同时维护三类模型状态：

- Actor/VLA policy：由 ActorGroup 更新。
- Rollout policy replica：由 RolloutGroup 做 no-grad inference，定期从 Actor 同步。
- World model + classifier：由 LearnerGroup 更新，定期同步给 WMEnvWorker。

Actor PPO 的训练数据来自 rollout 组装出的 trajectory，不应隐式从 replay 替代。

## 4. Eval

最终质量以真实 LIBERO eval 为准。WMEnv reward 可以提供训练信号，但不能替代真实环境 success rate。

## Data Flow

```text
Real LIBERO collection
  -> reward/hidden shards + manifest
  -> OnlineReplay / warmup datasets
  -> WM + classifier warmup checkpoints
  -> online cotrain
  -> actor/wm/classifier checkpoints + metrics
  -> real LIBERO eval
```

## Main Configs

- Launcher config：`configs/scripts/coldstart_warmup_cotrain.yaml`
- Sync cotrain experiment：`online_cotrain_pipeline_oft_backbone_latent`
- Manual Ray experiment：`manual_cotrain_ray_oft_backbone_latent`
- Eval config：`configs/scripts/eval_libero_vla.yaml`
