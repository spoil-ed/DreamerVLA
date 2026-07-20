# 2026-07-20 激进 Dreamer 实验执行说明

## 1. 进入环境

```bash
export DVLA_ROOT=/path/to/DreamerVLA
export DVLA_DATA_ROOT=/path/to/dreamervla-data

cd "${DVLA_ROOT}"
conda activate dreamervla
```

## 2. 获取旧实验选中的 WM/CLS checkpoint

```bash
export WM_CKPT="$(jq -r '.stages.world_model.selected_checkpoint' \
  "${DVLA_DATA_ROOT}/reproduction/manifests/training_state.json")"

export CLS_CKPT="$(jq -r '.stages.classifier.selected_checkpoint' \
  "${DVLA_DATA_ROOT}/reproduction/manifests/training_state.json")"

printf 'WM_CKPT=%s\nCLS_CKPT=%s\n' "${WM_CKPT}" "${CLS_CKPT}"
test -e "${WM_CKPT}"
test -e "${CLS_CKPT}"
```

优先使用上面得到的具体 `.ckpt` 文件。不要传入包含多个 checkpoint 的普通
`checkpoints/` 目录。

## 3. 启动前 dry-run

```bash
bash scripts/reproduce/02_train_dreamer.sh \
  --config reproduce/train_dreamer_aggressive \
  --wm_ckpt "${WM_CKPT}" \
  --cls_ckpt "${CLS_CKPT}" \
  dry_run=true
```

输出中应包含：

```text
--config openvla_libero_aggressive
manual_cotrain.global_steps=20
ngpu=8
```

## 4. 正式启动

```bash
bash scripts/reproduce/02_train_dreamer.sh \
  --config reproduce/train_dreamer_aggressive \
  --wm_ckpt "${WM_CKPT}" \
  --cls_ckpt "${CLS_CKPT}"
```

## 5. Docker 启动

先设置新版本的不可变镜像标签和容器内 checkpoint 路径：

```bash
export DREAMERVLA_IMAGE=spoil/dreamervla:sha-REPLACE_WITH_12_CHAR_COMMIT
export WM_CKPT_IN_CONTAINER=/data/outputs/reproduction/libero_goal/world_model/checkpoints/SELECTED_WM.ckpt
export CLS_CKPT_IN_CONTAINER=/data/outputs/reproduction/libero_goal/classifier/checkpoints/SELECTED_CLS.ckpt

read -rsp "W&B API key: " WANDB_API_KEY
export WANDB_API_KEY
```

启动容器：

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=100g \
  --ulimit memlock=-1 \
  --env WANDB_API_KEY \
  --volume "$PWD/dreamervla-data:/data" \
  "${DREAMERVLA_IMAGE}" \
  bash scripts/reproduce/02_train_dreamer.sh \
    --config reproduce/train_dreamer_aggressive \
    --wm_ckpt "${WM_CKPT_IN_CONTAINER}" \
    --cls_ckpt "${CLS_CKPT_IN_CONTAINER}"
```

Docker 中必须使用 `/data/...` 形式的容器内 checkpoint 路径。

## 6. 中断后续训

重复执行第 4 节或第 5 节的同一条命令。

## 7. 输出位置

```text
${DVLA_DATA_ROOT}/outputs/reproduction/libero_goal/openvla_libero_aggressive/dreamer/
${DVLA_DATA_ROOT}/reproduction/manifests/training_state_aggressive.json
```

## 8. W&B 检查

每次 eval 检查以下指标：

```text
eval/success_rate
eval/wm_trajectory_cosine
eval/cls_trajectory_f1
eval/cls_trajectory_accuracy
```
