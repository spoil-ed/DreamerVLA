# One-Trajectory DreamerVLA Config Map

本文只记录 one-trajectory SFT 到 DreamerVLA 的 config 对应关系。不是完整教程。

## 结论

目前没有一个专门叫 `openvla_oft_one_traj_to_dreamervla` 的一键全链路 config。

但是 OpenVLA-OFT 有逐阶段对应配置，可以手动串起来：

| 阶段 | OpenVLA-OFT config / 入口 | 说明 |
| --- | --- | --- |
| one-trajectory SFT | `openvla_oft_hdf5_one_trajectory` | 继承 `openvla_oft_hdf5` |
| action-hidden sidecar | `scripts/preprocess/preprocess_oft_action_hidden.py` | 这是脚本，不是 Hydra config |
| chunk world model | `oft_world_model_dinowm_chunk` | OpenVLA-OFT hidden 的 DINO-WM chunk 版 |
| latent classifier | `oft_latent_classifier_chunk` | OpenVLA-OFT chunk classifier |
| DreamerVLA / WMPO | `dreamervla_oft_dino_wm_wmpo_outcome` | OpenVLA-OFT + DINO-WM + WMPO outcome |
| eval | `eval_libero_vla` | 评估入口 |

注意：当前 `openvla_oft_hdf5_one_trajectory` 默认是 LM-head action-token SFT：

```yaml
policy.use_l1_regression: false
```

而 OFT DreamerVLA 链路默认期待 `oft_l1_regression` action hidden。若要把 one-trajectory OFT SFT 接到后续 WM/DreamerVLA，建议训练时覆盖：

```bash
policy.use_l1_regression=true
```

## RynnVLA One-Trajectory 主线

| 阶段 | config / 入口 |
| --- | --- |
| one-trajectory SFT | `vla_sft_one_trajectory` |
| action-hidden sidecar | `scripts/preprocess/prepare_libero_data.sh` |
| chunk world model | `world_model_dinowm_chunk` |
| latent classifier | `latent_classifier_libero_goal_chunk` |
| DreamerVLA / WMPO | `dreamervla_rynn_dino_wm_wmpo_outcome` |
| eval | `eval_libero_vla` |

下面命令里的 `<...>` 路径 override 都是可选的；不用显式路径时删掉对应行，让 task config 走默认值。

```bash
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh \
  task=libero_goal \
  init.vla_ckpt_path=<base_rynnvla_ckpt>

TASK=libero_goal \
VLA_CKPT=<one_traj_rynnvla_ckpt> \
HIDDEN_DIR=<one_traj_rynnvla_action_hidden_dir> \
bash scripts/preprocess/prepare_libero_data.sh

CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh \
  task=libero_goal \
  task.vla_ckpt_path=<one_traj_rynnvla_ckpt> \
  task.rynnvla_action_hidden_dir=<one_traj_rynnvla_action_hidden_dir>

CONFIG=latent_classifier_libero_goal_chunk bash scripts/train_wm.sh \
  task=libero_goal \
  data.success_dir_hidden=<one_traj_rynnvla_action_hidden_dir> \
  data.failure_dir_hidden=<rynnvla_failure_action_hidden_dir>

CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh \
  task=libero_goal \
  init.vla_ckpt_path=<one_traj_rynnvla_ckpt> \
  task.vla_ckpt_path=<one_traj_rynnvla_ckpt> \
  task.rynnvla_action_hidden_dir=<one_traj_rynnvla_action_hidden_dir> \
  init.world_model_state_ckpt=<rynnvla_wm_ckpt> \
  init.classifier_state_ckpt=<rynnvla_classifier_ckpt>

CONFIG=eval_libero_vla bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=<dreamervla_ckpt> \
  init.vla_ckpt_path=<one_traj_rynnvla_ckpt>
```

## OpenVLA-OFT One-Trajectory 链路

逐阶段 config 是齐的，但需要手动把上游产物通过 override 接给下游。

### 1. one-trajectory SFT

```bash
CONFIG=openvla_oft_hdf5_one_trajectory bash scripts/train_vla.sh \
  task=libero_goal \
  policy.use_l1_regression=true \
  task.openvla_oft.ckpt_path=<base_oft_ckpt_or_dir> \
  task.openvla_oft.component_ckpt_dir=<base_oft_component_dir> \
  task.openvla_oft.resume_step=<base_oft_resume_step>
```

产物一般在：

```text
data/outputs/vla/openvla_oft_lm_head_one_trajectory/<task>/<run>/
```

如果使用 `policy.use_l1_regression=true`，后续应使用该 run 保存的 OFT component checkpoint。

### 2. 重新抽 OpenVLA-OFT action-hidden sidecar

```bash
python scripts/preprocess/preprocess_oft_action_hidden.py \
  --hdf5-dir <raw_hdf5_dir> \
  --oft-ckpt <one_traj_oft_ckpt_or_component_dir> \
  --out-action-dir <one_traj_oft_action_hidden_dir> \
  --skip-cd-sidecars \
  --overwrite
```

这一阶段没有 Hydra config；它负责生成后续 WM / classifier / DreamerVLA 使用的 hidden sidecar。

### 3. 训练 OFT chunk WM

```bash
CONFIG=oft_world_model_dinowm_chunk bash scripts/train_wm.sh \
  task=libero_goal \
  task.openvla_oft.ckpt_path=<one_traj_oft_ckpt_or_component_dir> \
  task.openvla_oft.component_ckpt_dir=<one_traj_oft_ckpt_or_component_dir> \
  task.openvla_oft.resume_step=<one_traj_oft_resume_step> \
  task.openvla_oft.action_hidden_dir=<one_traj_oft_action_hidden_dir>
```

### 4. 训练 OFT classifier

```bash
CONFIG=oft_latent_classifier_chunk bash scripts/train_wm.sh \
  task=libero_goal \
  task.openvla_oft.action_hidden_dir=<one_traj_oft_action_hidden_dir> \
  task.openvla_oft.failure_action_hidden_dir=<oft_failure_action_hidden_dir>
```

### 5. 训练 OFT DreamerVLA

```bash
CONFIG=dreamervla_oft_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh \
  task=libero_goal \
  task.openvla_oft.ckpt_path=<one_traj_oft_ckpt_or_component_dir> \
  task.openvla_oft.component_ckpt_dir=<one_traj_oft_ckpt_or_component_dir> \
  task.openvla_oft.resume_step=<one_traj_oft_resume_step> \
  task.openvla_oft.action_head_ckpt=<one_traj_oft_action_head_ckpt> \
  task.openvla_oft.action_hidden_dir=<one_traj_oft_action_hidden_dir> \
  init.world_model_state_ckpt=<oft_wm_ckpt> \
  init.classifier_state_ckpt=<oft_classifier_ckpt>
```

评估 DreamerVLA ckpt 时同样显式给 ckpt：

```bash
CONFIG=eval_libero_vla bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=<dreamervla_ckpt>
```

## 常用 task override

```bash
task=libero_goal
task=libero_object
task=libero_spatial
task=libero_10
```

task YAML 中已有 OpenVLA-OFT 默认 ckpt / sidecar 路径；如果要使用 one-trajectory 产物，必须覆盖：

```bash
task.openvla_oft.ckpt_path=...
task.openvla_oft.component_ckpt_dir=...
task.openvla_oft.resume_step=...
task.openvla_oft.action_head_ckpt=...
task.openvla_oft.action_hidden_dir=...
```
