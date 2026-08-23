# 2026-07-20 激进 Dreamer Docker 实验执行说明

## 1. 拉取 Docker 镜像

```bash
docker pull spoil/dreamervla:v1.1
```

Docker 只下载本机缺少或发生变化的层。日志中的 `Layer already exists` 表示该层直接
复用，不会重新下载整个镜像。

拉取 `v1.1` 不会修改旧镜像、旧容器或挂载在 `/data` 的实验数据。

## 2. 设置数据目录和 checkpoint

进入包含 `dreamervla-data/` 的宿主机目录：

```bash
cd /path/to/docker-run-directory
export DREAMERVLA_DATA="$PWD/dreamervla-data"
```

从旧实验状态文件读取容器内 WM/CLS checkpoint 路径：

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

两个变量应为 `/data/.../*.ckpt`。不要传入包含多个文件的普通 `checkpoints/` 目录。

## 3. 设置 W&B

```bash
read -rsp "W&B API key: " WANDB_API_KEY
export WANDB_API_KEY
```

## 4. 启动前 dry-run

```bash
docker run --rm \
  --ipc=host \
  --network=host \
  --shm-size=100g \
  --ulimit memlock=-1 \
  --env WANDB_API_KEY \
  --volume "${DREAMERVLA_DATA}:/data" \
  spoil/dreamervla:v1.1 \
  bash scripts/reproduce/02_train_dreamer.sh \
    --config reproduce/train_dreamer_aggressive \
    --wm_ckpt "${WM_CKPT_IN_CONTAINER}" \
    --cls_ckpt "${CLS_CKPT_IN_CONTAINER}" \
    dry_run=true
```

输出应包含：

```text
--config openvla_libero_aggressive
manual_cotrain.global_steps=20
ngpu=8
```

## 5. 正式启动

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=100g \
  --ulimit memlock=-1 \
  --env WANDB_API_KEY \
  --volume "${DREAMERVLA_DATA}:/data" \
  spoil/dreamervla:v1.1 \
  bash scripts/reproduce/02_train_dreamer.sh \
    --config reproduce/train_dreamer_aggressive \
    --wm_ckpt "${WM_CKPT_IN_CONTAINER}" \
    --cls_ckpt "${CLS_CKPT_IN_CONTAINER}"
```

## 6. 中断后续训

重新执行第 5 节的同一条命令。

## 7. 输出位置

宿主机：

```text
dreamervla-data/outputs/reproduction/libero_goal/openvla_libero_aggressive/dreamer/
dreamervla-data/reproduction/manifests/training_state_aggressive.json
```

容器内：

```text
/data/outputs/reproduction/libero_goal/openvla_libero_aggressive/dreamer/
/data/reproduction/manifests/training_state_aggressive.json
```

## 8. W&B 检查

```text
eval/success_rate
eval/wm_trajectory_cosine
eval/cls_trajectory_f1
eval/cls_trajectory_accuracy
```
