# 2026-07-20 Imagined-Success SFT 训练信号实验执行说明

这个实验只回答一个最小问题：已有 world model 和 success classifier 产生的
imagined trajectories，能否为基础 OpenVLA policy 提供真实、可提交的训练信号。

固定流程为：收集 1 条 real episode 作为起点，使用冻结 WM/CLS 想象 128 条最长
512 个物理 step 的轨迹，按 CLS checkpoint 自带的 threshold 选择成功轨迹，然后只对
成功轨迹执行 1 次 action-token SFT。实验不会预训练或更新 WM、CLS、encoder，也不会
计算 advantage 或执行 PPO。

## 1. 准备包含探针代码的 Docker 镜像

该实验要求镜像包含 DreamerVLA commit `12e1354` 或更新版本。已有
`spoil/dreamervla:v1.1` 不包含本实验配置，不要直接使用它启动探针。

在当前 DreamerVLA checkout 中构建镜像：

```bash
cd /path/to/DreamerVLA

docker build \
  --file docker/Dockerfile \
  --build-arg DVLA_GIT_COMMIT="$(git rev-parse HEAD)" \
  --build-arg DVLA_IMAGE_VERSION=success-sft-probe \
  --build-arg DVLA_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --tag dreamervla:success-sft-probe \
  .

export DREAMERVLA_IMAGE=dreamervla:success-sft-probe
```

如果之后已有包含该 commit 的发布镜像，可以把 `DREAMERVLA_IMAGE` 换成对应的固定
tag 或 digest。检查镜像内记录的源码版本：

```bash
docker run --rm "${DREAMERVLA_IMAGE}" \
  python -c 'import json; print(json.load(open(".dreamervla-image.json"))["commit"])'
```

## 2. 设置数据目录和 checkpoint

进入包含 `dreamervla-data/` 的宿主机目录：

```bash
cd /path/to/docker-run-directory
export DREAMERVLA_DATA="$PWD/dreamervla-data"
```

从已有 reproduction 状态读取容器内 WM/CLS checkpoint 路径：

```bash
export WM_CKPT_IN_CONTAINER="$(jq -r '.stages.world_model.selected_checkpoint' \
  "${DREAMERVLA_DATA}/reproduction/manifests/training_state.json")"

export CLS_CKPT_IN_CONTAINER="$(jq -r '.stages.classifier.selected_checkpoint' \
  "${DREAMERVLA_DATA}/reproduction/manifests/training_state.json")"

printf 'WM=%s\nCLS=%s\n' \
  "${WM_CKPT_IN_CONTAINER}" \
  "${CLS_CKPT_IN_CONTAINER}"
```

检查宿主机上的对应文件：

```bash
test -e "${DREAMERVLA_DATA}/${WM_CKPT_IN_CONTAINER#/data/}"
test -e "${DREAMERVLA_DATA}/${CLS_CKPT_IN_CONTAINER#/data/}"
```

两个变量应为 `/data/.../*.ckpt`。必须同时传入 WM 和 CLS checkpoint，不要传入包含
多个文件的普通 `checkpoints/` 目录。CLS checkpoint 必须保存训练时选定的
`classifier_threshold`；探针不会自行降低或重新拟合 threshold。

## 3. 启动前 dry-run

该探针只记录 TensorBoard，不需要设置 `WANDB_API_KEY`。

```bash
docker run --rm \
  --ipc=host \
  --network=host \
  --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "${DREAMERVLA_DATA}:/data" \
  "${DREAMERVLA_IMAGE}" \
  bash scripts/reproduce/02_train_dreamer.sh \
    --config reproduce/train_dreamer_success_sft_probe \
    --wm_ckpt "${WM_CKPT_IN_CONTAINER}" \
    --cls_ckpt "${CLS_CKPT_IN_CONTAINER}" \
    dry_run=true
```

输出应包含：

```text
--config openvla_libero_success_sft_probe
manual_cotrain.global_steps=1
ngpu=8
```

有效 Hydra 配置还应满足：

```text
manual_cotrain.training_mode=imagined_success_sft
manual_cotrain.real_rollout_target_trajectories=1
manual_cotrain.wm_rollout_target_trajectories=128
manual_cotrain.max_steps_per_rollout_epoch=512
manual_cotrain.require_training_signal=true
actor.train_cfg.global_batch_size=8192
```

这里 8192 是 `128 trajectories * (512 / 8 action-chunk size)` 的 nominal
trajectory-decision 数。轨迹在达到 CLS success threshold 后可以提前终止。

## 4. 正式启动

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=100g \
  --ulimit memlock=-1 \
  --volume "${DREAMERVLA_DATA}:/data" \
  "${DREAMERVLA_IMAGE}" \
  bash scripts/reproduce/02_train_dreamer.sh \
    --config reproduce/train_dreamer_success_sft_probe \
    --wm_ckpt "${WM_CKPT_IN_CONTAINER}" \
    --cls_ckpt "${CLS_CKPT_IN_CONTAINER}"
```

严格的阶段顺序是：

1. 使用基础 VLA 收集 1 条完整 real episode；
2. 将 episode start 写入 replay，冻结 encoder、WM 和 CLS；
3. 生成 128 条最长 512-step imagined trajectories；
4. 选择任一有效时刻 CLS score 达到 checkpoint threshold 的完整轨迹；
5. 仅在这些轨迹的有效 action decisions 上最小化 action-token NLL；
6. 在 `max_policy_kl=0.05` 的事务内提交或回滚 actor update；
7. 写入 checkpoint，并自动判断训练信号是否成立。

## 5. 通过条件

程序只有同时满足下列条件才会正常结束：

- `actor/success_sft_trajectories >= 1`；
- `actor/success_sft_valid_samples >= 1`；
- `actor/success_sft_grad_norm` 为有限正数；
- `actor/success_sft_optimizer_steps >= 1`；
- `actor/success_sft_update_committed == 1`，即 KL 事务没有回滚；
- `applied_policy_steps >= 1`；
- checkpoint 中的 `policy_initial_hash` 与 `policy_final_hash` 不同。

最终 TensorBoard 指标 `train/training_signal_passed=1` 表示上述条件全部成立。它只证明
当前 WM/CLS imagined data 对 actor 存在可执行梯度，不代表 real LIBERO success rate
已经提高。

## 6. 输出位置

宿主机：

```text
dreamervla-data/outputs/reproduction/libero_goal/openvla_libero_success_sft_probe/dreamer/
dreamervla-data/reproduction/manifests/training_state_success_sft_probe.json
```

容器内：

```text
/data/outputs/reproduction/libero_goal/openvla_libero_success_sft_probe/dreamer/
/data/reproduction/manifests/training_state_success_sft_probe.json
```

查看 TensorBoard：

```bash
tensorboard --logdir \
  "${DREAMERVLA_DATA}/outputs/reproduction/libero_goal/openvla_libero_success_sft_probe/dreamer/tensorboard"
```

## 7. 离线复核 checkpoint

不加载 VLA、WM 或 CLS，即可从 checkpoint metrics 和 policy hash 重新判断：

```bash
docker run --rm \
  --volume "${DREAMERVLA_DATA}:/data" \
  "${DREAMERVLA_IMAGE}" \
  python -m dreamervla.diagnostics.verify_training_signal \
    /data/outputs/reproduction/libero_goal/openvla_libero_success_sft_probe/dreamer
```

通过时命令输出类似下面的 JSON：

```json
{
  "checkpoint": "/data/outputs/reproduction/libero_goal/openvla_libero_success_sft_probe/dreamer/checkpoints/latest.ckpt",
  "passed": true,
  "failures": [],
  "evidence": {
    "applied_policy_steps": 1,
    "grad_norm": 0.0123,
    "optimizer_steps": 1,
    "policy_final_hash": "<sha256-after>",
    "policy_initial_hash": "<sha256-before>",
    "successful_imagined_trajectories": 1,
    "training_mode": "imagined_success_sft",
    "valid_sft_samples": 64
  }
}
```

数值取决于实际 rollout。失败时 `passed=false`，`failures` 给出具体原因，进程退出码为 1。

## 8. 中断后恢复

reproduction 状态和 run root 与 aggressive 实验隔离：

```text
training_state_success_sft_probe.json
openvla_libero_success_sft_probe/dreamer/
```

如果进程在写入最终 checkpoint 之前意外中断，重新执行第 4 节的同一条命令会从原 run
root 恢复。已经完整通过的 stage 会经过 checkpoint 校验后跳过。

如果程序明确报告 training-signal failure，不要把它当作普通中断后盲目续跑：一步实验的
checkpoint 此时可能已经到达目标 step。先用第 7 节命令保存失败 evidence；修正 checkpoint
或配置后，将原 probe run root 和 state manifest 移到归档位置，再启动一次全新的 probe。

## 9. 常见失败

- `classifier selected no successful imagined trajectory`：128 条轨迹都没有达到 checkpoint
  threshold。这本身就是探针的有效失败结论；不要自动降低 threshold 或把失败轨迹加入 SFT。
- `successful imagined trajectories contained no valid SFT sample`：成功轨迹缺少有效
  `loss_mask` action decisions，应检查 rollout/action-token sidecar 契约。
- `actor gradient norm was not finite and positive`：检查 action-token labels、log-prob 和
  actor trainable parameter partition；不要用零梯度 optimizer step 冒充训练信号。
- `policy update was not committed`：行为 KL 超过 `0.05`，更新已回滚。先检查学习率、样本
  分布和旧策略 log-prob，不要直接关闭 KL transaction。
- `policy hash did not change`：checkpoint 没有观察到实际 actor 参数变化。
- CUDA/Ray handshake timeout：保留 run root，读取 rank-0 log 和
  `diagnostics/manual_cotrain_progress/current/`，确认 real rollout、WM imagination 和 ActorGroup
  SFT 按顺序推进。
