# OpenVLA_Onetraj_LIBERO Pipeline

Goal: run the OpenVLA-OFT one-trajectory + LIBERO-Goal pipeline with matching
Hydra task and processed-data artifact names.

Canonical task name:

```text
OpenVLA_Onetraj_LIBERO
```

This writes intermediate data under:

```text
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/
```

The raw benchmark suite is still `libero_goal`; `OpenVLA_Onetraj_LIBERO` is the
pipeline task name and `OpenVLA_Onetraj_LIBERO_libero_goal` is the
preprocessing artifact name.

Current OpenVLA-OFT DreamerVLA implementation keeps one action-probability
route for the downloaded one-trajectory checkpoint:

- Discrete token: use the OpenVLA LM head over action-token slots, sample
  OpenVLA action token IDs, and decode those IDs to continuous actions with
  the OpenVLA action-token bin mapping.

TODO(agent): add and validate the OpenVLA-OFT L1 Gaussian route separately.
That route needs a real `action_head--*_checkpoint.pt`, decodes action-hidden
slots to continuous means, and learns the diagonal Gaussian std. Do not treat
the downloaded one-trajectory HF checkpoint as that route; it has no L1 action
head.

If the goal is a one-trajectory baseline without depending on OpenVLA-OFT L1
components, train a one-trajectory RynnVLA checkpoint instead. That path writes
a Hugging Face sidecar (`latest_hf/`) and uses the RynnVLA action head in the
regular RynnVLA pipeline.

## 0. System

```bash
cd /path/to/DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"

bash scripts/install_env.sh
conda activate dreamervla
```

## 1. Download

Download LIBERO-Goal and one-trajectory OpenVLA-OFT assets:

```bash
bash scripts/download_assets.sh \
  download.rynnvla=false \
  download.libero=true \
  download.openvla_one_traj=true \
  env.LIBERO_SUITES=[libero_goal] \
  only=[30_openvla_oft_one_trajectory,40_libero_dataset]
```

The downloaded one-trajectory checkpoint path is:

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1
```

That downloaded checkpoint is a discrete OpenVLA-OFT checkpoint. It has no
`action_head--*_checkpoint.pt`, so do not run it with `OFT_POLICY_MODE=l1`.
The L1/Gaussian DreamerVLA chain is a TODO, not the default path in this
tutorial.

Alternative one-trajectory RynnVLA baseline:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=RynnVLA_LIBERO \
  libero_suite=libero_goal \
  only=[20_pretokenize_dataset] \
  gpus=0 ngpu=1
```

```bash
bash scripts/train_vla.sh \
  experiment=vla_sft_one_trajectory \
  task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

The one-trajectory RynnVLA SFT route reads
`${DVLA_DATA_ROOT}/configs/RynnVLA_LIBERO_libero_goal/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml`
and filters each split to one selected `task/trj_*` trajectory.

Use the produced HF directory for RynnVLA hidden extraction and eval:

```bash
VLA_CKPT=/abs/path/to/rynnvla_run/checkpoints/latest_hf \
bash scripts/preprocess/prepare_libero_data.sh \
  task=RynnVLA_LIBERO \
  libero_suite=libero_goal \
  only=[30_action_hidden] \
  gpus=0 ngpu=1 \
  env.VLA_CKPT="${VLA_CKPT}"
```

## 2. Preprocess

OpenVLA-OFT Scheme A does not use `20_pretokenize_dataset`. That step builds
RynnVLA token-record configs for tokenized VLA SFT and older pretokenized
dataset routes. The OFT action-hidden WM/DreamerVLA path only needs the
reward-labeled HDF5 from `10_hdf5_reward` plus the OFT action-hidden sidecar
from `35_oft_action_hidden`.

Build the reward HDF5:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal \
  only=[10_hdf5_reward] \
  gpus=0 ngpu=1
```

Extract OpenVLA-OFT action-hidden Scheme A sidecars in discrete mode. This loads
the merged LM-head checkpoint directly and leaves `action_head=None`; the
hidden states still come from the OpenVLA-OFT backbone layer, but there are no
L1 C/D action-head intermediates:

```bash
OFT_DISCRETE_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal \
  only=[35_oft_action_hidden] \
  gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=action_hidden \
  env.OFT_POLICY_MODE=discrete \
  env.OFT_HISTORY=1 \
  env.OFT_IMAGE_KEYS=agentview_rgb \
  env.OFT_CKPT="${OFT_DISCRETE_CKPT}"
```

The classifier does not introduce another action-probability scheme. It is
compatible with action-hidden sidecars generally; for this discrete-token
OpenVLA path, point the existing OpenVLA classifier route at the same `h1`
sidecar family used by WM and DreamerVLA.

Expected artifacts:

```text
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1
```

Manual integrity check:

```bash
python -m dreamervla.preprocess.check_artifacts hdf5-dir \
  --dir "${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  --reference-dir "${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256" \
  --match-reference-demos \
  --match-reference-lengths

python -m dreamervla.preprocess.check_artifacts hdf5-dir \
  --dir "${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1" \
  --reference-dir "${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  --match-reference-demos \
  --match-reference-lengths \
  --require-complete-attr \
  --require-config
```

This checks that the reward HDF5 files match the no-noops source files and that
the OFT action-hidden sidecar has the same file set, demo keys, per-demo
lengths, `complete=true` markers, and `preprocess_config.json` schema metadata.

If `.tmp` or `.rank*.tmp` files remain under the artifact directories, the usual
reason is that preprocessing was interrupted before the atomic rename to the
final `.hdf5` completed. Re-running the same preprocessing step removes the old
rank-local tmp for that output before writing it again. Only delete tmp files by
hand after confirming no preprocessing process is still running:

```bash
find "${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal" \
  -type f \( -name "*.tmp" -o -name "*.rank*.tmp" \) -print
```

## 3. World Model

```bash
bash scripts/train_wm.sh \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=4
```

Add logging overrides to any training command. The project uses TensorBoard
event files for the local TensorFlow-compatible log viewer:

```bash
logger=tensorboard
logger=wandb
logger=tensorboard_wandb runner.logger.wandb_mode=online
logger=tensorboard_wandb runner.logger.wandb_mode=offline
```

TensorBoard writes `${training.out_dir}/log/tensorboard`; W&B writes
`${training.out_dir}/log/wandb`. `wandb_mode=offline` keeps the W&B run local for
later sync.

Smoke run:

```bash
bash scripts/train_wm.sh \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=1 num_workers=0 max_steps=1 \
  out_dir=/tmp/openvla_onetraj_libero_discrete_wm_smoke
```

## 4. Classifier

WMPO needs failure rollout HDF5 files and matching OFT failure sidecars.

```bash
bash scripts/train_wm.sh \
  experiment=oft_latent_classifier_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 batch_size=8 num_workers=4 \
  -- \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1" \
  task.openvla_oft.expected_action_head_type=oft_discrete_token \
  task.openvla_oft.expected_include_state=false \
  task.openvla_oft.expected_history=1 \
  task.openvla_oft.failure_hdf5_dir=/abs/path/to/OpenVLA_Onetraj_LIBERO_libero_goal_failures \
  task.openvla_oft.failure_action_hidden_dir=/abs/path/to/OpenVLA_Onetraj_LIBERO_libero_goal_failures_oft_discrete_action_hidden
```

## 5. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_oft_discrete_token_dino_wm_wmpo_outcome \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 \
  -- \
  init.world_model_state_ckpt=/abs/path/to/oft_discrete_token_world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/openvla_action_hidden_classifier.ckpt
```

## 6. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/openvla_onetraj_dreamervla.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

Raw OpenVLA-OFT checkpoint eval:

```bash
CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
SUITE=libero_goal \
GPU_ID=0 \
POLICY_MODE=discrete \
CAMERA_INPUTS=primary \
NUM_IMAGES=1 \
USE_PROPRIO=0 \
bash scripts/eval/launch_openvla_oft_official_libero_eval.sh
```
