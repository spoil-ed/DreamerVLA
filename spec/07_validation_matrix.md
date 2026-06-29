# Validation Matrix

状态：current

本文把验证命令映射到它们证明的契约，并区分 config/unit 覆盖与 runtime/long-run 证据。

## Unit Contracts

| 契约 | 测试 |
| --- | --- |
| public runner export 与 route composition | `test_runner_public_api.py`、`test_manual_cotrain_ray_runner.py` |
| manual placement（0–5 GPU） | `test_manual_cotrain_placement.py`、`test_scheduler_placement.py` |
| manual resources / FSDP / chunk divisibility 校验 | `test_manual_cotrain_config_validation.py`、`test_config_validation.py` |
| typed trajectory messages 与 chunk-level collation | `test_cotrain_messages.py` |
| rollout no-grad policy copy 与 OFT encoding | `test_multistep_rollout_worker.py`、`test_oft_rollout_bundle.py` |
| EnvWorker chunk stepping、replay sidecar、EGL recovery、WM sync | `test_trajectory_env_worker.py`、`test_env_worker_spawn_recovery.py`、`test_env_worker_world_model_sync.py`、`test_env_worker_record_builder.py` |
| Actor PPO shape 与 FSDP-safe sync | `test_embodied_fsdp_actor.py` |
| LearnerGroup WM/classifier-only 模式与版本/精度 | `test_learner_worker_manual_precision.py`、`test_learner_worker_component_versions.py` |
| WMEnv replay bootstrap | `test_wm_env_bootstrap.py`、`test_replay_client_sample_forwarding.py`、`test_latent_world_model_env.py` |
| cold-start launcher 与 warmup bridge | `test_coldstart_warmup_cotrain_launcher.py`、`test_ray_init_ckpt_warmup_bridge.py` |

> 用 `dreamervla` conda 环境（py3.11 / transformers 4.40.1）跑测试；base 环境会有约 13 个伪失败。

focused 本地命令：

```bash
PYTHONPATH=$PWD pytest \
  tests/unit_tests/test_manual_cotrain_placement.py \
  tests/unit_tests/test_cotrain_messages.py \
  tests/unit_tests/test_multistep_rollout_worker.py \
  tests/unit_tests/test_trajectory_env_worker.py \
  tests/unit_tests/test_embodied_fsdp_actor.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_manual_cotrain_config_validation.py \
  tests/unit_tests/test_wm_env_bootstrap.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py \
  -q
```

## Tiny Runtime Smoke

CPU/manual smoke（`ngpu=0`）：

```bash
WANDB_MODE=offline PYTHONPATH=$PWD \
python -m dreamervla.train \
  experiment=manual_cotrain_ray_tiny \
  training.out_dir=/tmp/dvla_manual_cotrain_tiny_smoke
```

证明 manual route 能启动、构建四个 group、完成一个短 global step 并退出。它不证明真实
OpenVLA-OFT/LIBERO 稳定性。

跑一次 learner update 的变体：

```bash
PYTHONPATH=$PWD python -m dreamervla.train \
  experiment=manual_cotrain_ray_tiny \
  training.out_dir=/tmp/dvla_manual_cotrain_tiny_learner \
  manual_cotrain.global_steps=1 \
  manual_cotrain.learner_update_step=1
```

预期 logged metrics 中出现 `env/trajectory_shards`、`actor/ppo_updates` 和 learner 更新计数，
进程以 code 0 退出。

## 0–5 GPU Startup Dry-Run

对 `N=0..5` 验证 launcher 命令（manual Ray route 不使用 `torchrun`）：

```bash
PYTHONPATH=$PWD python -m dreamervla.launchers.coldstart_warmup_cotrain \
  mode=ray task=goal ngpu=N profile=smoke \
  cotrain_engine=async cotrain_phase=online dry_run=true
```

每个 N 的预期输出包含 `experiment=manual_cotrain_ray_oft_backbone_latent` 和
`manual_cotrain.ngpu=N`。

## GPU/LIBERO Runtime

async 主线 dry run：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu cotrain_engine=async render_backend=egl dry_run=true
```

async 主线短 run：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu cotrain_engine=async render_backend=egl \
  collect.episodes_per_task=2 warmup.total_env_steps=1024
```

完整验证仍需要带 LIBERO assets、OpenVLA-OFT checkpoint 和稳定 EGL/MuJoCo 的目标机器。通过 unit
test 和 tiny CPU smoke 不足以宣称完整 cotrain 成功。

## Failure Triage

- config compose 失败：检查 `configs/experiment/*`、`configs/dreamervla/*` 和
  `dreamervla.config.validate_cfg`。
- channel deadlock：检查 EnvGroup/RolloutGroup 的 `key`（`<env_rank>:<slot_id>`）对齐和 `StopMsg`
  投递。
- FSDP sync hang：确认 ActorGroup 在所有 rank 上调 state export。
- 缺 sidecar：检查 `RolloutResultMsg.forward_inputs` 与 EnvWorker replay transition 组装。
- WMEnv bad reset：检查 replay bootstrap key 和 `LatentWorldModelEnv` slot 状态。
- EGL abort：检查 `env.real.cfg.egl_*`、`MUJOCO_EGL_DEVICE_ID` 和 spawn-child 日志。
