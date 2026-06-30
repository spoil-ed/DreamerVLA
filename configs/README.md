# Config Registry

DreamerVLA uses the standard Hydra template pattern:

```text
configs/
├── train.yaml
├── experiment/
├── VLA/
├── worldmodel/
├── classifier/
├── dreamervla/
├── evaluation/
├── task/
└── logger/
```

`train.yaml` is the stable entrypoint. `experiment=<name>` selects a recipe
under `configs/experiment/`; that recipe overrides one cohesive module group
(`VLA`, `worldmodel`, `classifier`, `dreamervla`, or `evaluation`). Keep shell
overrides operational and small: GPUs, batch size, checkpoint paths, output
tags, and smoke-test limits.

`logger=tensorboard_wandb` is the default for grouped training. It writes local
TensorBoard event files under `${training.out_dir}/log/tensorboard` and W&B run
files under `${training.out_dir}/log/wandb`. W&B defaults to online mode; set
`runner.logger.wandb_mode=offline` for local-only W&B logs:

```bash
python -m dreamervla.train experiment=world_model_chunk
python -m dreamervla.train experiment=world_model_chunk runner.logger.wandb_mode=offline
python -m dreamervla.train experiment=world_model_chunk logger=tensorboard
python -m dreamervla.train experiment=world_model_chunk logger=wandb
```

Optional Ray backend resource knobs live in explicit config groups:

| Group | Options | Purpose |
| --- | --- | --- |
| `precision` | `fp32`, `bf16`, `fp16` | Manual learner AMP precision |
| `parallelism` | `none`, `fsdp` | Manual learner FSDP / CPU-offload / checkpointing knobs |
| `scheduler` | `local`, `ray_auto` | Single-node Ray cluster and component-placement metadata |

Example single-node multi-GPU Ray placement without FSDP:

```bash
CUDA_VISIBLE_DEVICES=2,3 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
python -m dreamervla.train \
  experiment=online_cotrain_ray_dreamervla_tiny \
  +inference.placement.strategy=packed \
  +inference.placement.gpu_id=0 \
  inference.cfg.device=auto \
  +learner.placement.strategy=packed \
  +learner.placement.start_gpu=1 \
  +learner.placement.end_gpu=1 \
  +learner.placement.num_gpus_per_worker=1 \
  learner.train_cfg.device=auto
```

This places inference on physical GPU 2 and learner on physical GPU 3. Inside
each actor the selected card is local `cuda:0`.

`+parallelism=fsdp` is separate from placement. It exposes manual learner-side
sharding knobs under
`learner.train_cfg.fsdp`, including `strategy=fsdp|fsdp2|no_shard`, AMP
precision, CPU offload, activation checkpointing, and process-group backend.
These knobs are single-node only; multi-node Ray placement is not a DreamerVLA
target.

Visualize TensorBoard logs from a run root with:

```bash
tensorboard --logdir "${OUT_DIR}/log/tensorboard" --host 0.0.0.0 --port 6006
```

If the run is on a remote machine, forward the port from your laptop, then open
`http://localhost:6006`:

```bash
ssh -L 6006:localhost:6006 user@host
```

For W&B online runs, open the run URL printed by `wandb` in the training log or
visit the project in the W&B web UI. For offline W&B runs, sync after training:

```bash
wandb sync "${OUT_DIR}/log/wandb"
```

Before a runner is instantiated, `dreamervla.config.validate_cfg` performs
RLinf-style lightweight checks: logger backend names, explicit resume paths,
actor-update route names from `dreamervla.algorithms.registry`, sidecar route
consistency, chunk / horizon compatibility, optional path existence via
`validation.require_existing_paths=true`, and any configured
`training.global_batch_size` divisibility by `WORLD_SIZE *
training.gradient_accumulate_every`.

Runtime artifacts should stay under one run root:

```text
${training.out_dir}/
├── resolved_config.yaml
├── run_manifest.json
├── checkpoints/
│   ├── latest.ckpt
│   └── global_step_<N>/
├── log/
│   ├── tensorboard/
│   └── wandb/
├── video/
│   ├── train/
│   └── eval/
└── diagnostics/
```

Older `ckpt/latest.ckpt` files remain load-compatible for resume, but new
default `BaseRunner.save_checkpoint()` writes to `checkpoints/latest.ckpt`.

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| VLA training | `scripts/train_vla.sh` | `vla_rynnvla_action_head` |
| VLA one-trajectory SFT | `scripts/train_vla.sh experiment=vla_sft_one_trajectory` | `vla_sft_one_trajectory` |
| OpenVLA-OFT one-trajectory SFT | `scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory` | `openvla_oft_hdf5_one_trajectory` |
| OpenVLA-OFT L1 one-trajectory SFT | `scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory_l1` | `openvla_oft_hdf5_one_trajectory_l1` |
| WM training | `scripts/train_wm.sh` | `world_model_chunk` |
| OpenVLA hidden_state WM training | `scripts/train_wm.sh experiment=oft_world_model_chunk` | `oft_world_model_chunk` |
| DreamerVLA cotrain pipeline | `scripts/train_dreamervla.sh` | `openvla_onetraj_libero_cotrain_noray` |
| LIBERO eval | `scripts/eval_libero_vla.sh` | `eval_libero_vla` |

## Experiments

| Experiment | Module group |
| --- | --- |
| `vla_rynnvla_action_head` | `VLA/rynnvla_action_head` |
| `vla_sft_one_trajectory` | `VLA/rynnvla_one_trajectory` |
| `openvla_oft_hdf5` | `VLA/openvla_oft` |
| `openvla_oft_hdf5_one_trajectory` | `VLA/openvla_oft_one_trajectory` |
| `openvla_oft_hdf5_one_trajectory_l1` | `VLA/openvla_oft_l1_one_trajectory` |
| `world_model_step` | `worldmodel/rynnvla_action_step` |
| `world_model_chunk` | `worldmodel/rynnvla_action_chunk` |
| `oft_world_model_chunk` | `worldmodel/openvla_oft_input_token_chunk` |
| `oft_discrete_token_world_model_chunk` | `worldmodel/openvla_oft_discrete_token_action_chunk` |
| `latent_classifier_libero_goal_chunk` | `classifier/rynnvla_action_chunk` |
| `oft_latent_classifier_chunk` | `classifier/openvla_oft_input_token_chunk` |
| `openvla_onetraj_libero_cotrain_noray` | `dreamervla/openvla_onetraj_libero_cotrain_noray` |
| `openvla_onetraj_libero_cotrain_ray` | `dreamervla/openvla_onetraj_libero_cotrain_ray` |
| `eval_libero_vla` | `evaluation/libero_vla` |

The earlier DreamerVLA LUMOS route aliases (`dreamervla_rynn_wm_lumos`,
`dreamervla_oft_wm_lumos`, and their token/discrete-token variants) were
consolidated into the `openvla_onetraj_libero_cotrain_*` mainline above; see
`docs/reference/routes.md` for the historical role-based mapping.

Module configs use Hydra defaults to include the task config:

```yaml
defaults:
  - _self_
  - /task: libero_goal
```

Keep concrete dataset task paths, horizons, sidecar expectations, and
task-specific dimensions in `task/*.yaml`. The module configs define the
runner, model, optimizer, and algorithm for that experiment family.

## Task Configs

The task folder contains concrete dataset task definitions only:

```text
task/libero_goal.yaml
task/libero_object.yaml
task/libero_spatial.yaml
task/libero_10.yaml
task/rynnvla_libero.yaml
task/openvla_onetraj_libero.yaml
task/openvla_onetraj_libero_object.yaml
task/openvla_onetraj_libero_spatial.yaml
task/openvla_onetraj_libero_10.yaml
task/openvla_onetraj_coldstart_libero*.yaml
```

Switch tasks with Hydra, for example:

```bash
bash scripts/train_vla.sh task=libero_object
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal
bash scripts/train_wm.sh experiment=world_model_chunk task=libero_spatial
bash scripts/train_wm.sh experiment=world_model_chunk task=rynnvla_libero
bash scripts/train_wm.sh experiment=oft_world_model_chunk task=openvla_onetraj_libero
```

OpenVLA-OFT DreamerVLA/WM defaults use WM query-before
`hidden_state` tensors from the `input_token_embedding` sidecar. Action-hidden
groups are legacy-only and are not the default route.

`rynnvla_libero` and `openvla_onetraj_libero` are pipeline task aliases over
the raw `libero_goal` benchmark suite. Their preprocessing artifact names append
the dataset suite, e.g. `OpenVLA_Onetraj_LIBERO_libero_goal` (artifact dirs keep
their historical names), and use matching
processed-data prefixes under `processed_data/<artifact>/<stage>`.
The OpenVLA one-trajectory matrix also has object, spatial, and libero_10 task
configs. Cold-start OpenVLA tasks inherit the matching one-trajectory VLA and
suite config, then reroot outputs under collected rollout directories.
