# π0.5 prefix-input world model

Start the eight-GPU `[968,2048]` prefix-input WM training run with:

```bash
bash experiments/pi05_wm/train_prefix_input.sh
```

The recipe uses global batch 128 (local batch 16 on each of eight ranks), 20,000
WM updates, learning rate `1e-4`, and exact language slots. Outputs are written
under `data/outputs/wm_pi05_prefix_input_train/<timestamp>/` unless `RUN_ROOT` is
set.

The launcher loads `/jfs/oss-import/xinglei/.secrets/wandb.env` without printing
its contents and logs to the W&B project `dreamer`.

This WM consumes DreamerVLA RGB rollout HDF5 shards, not the LeRobot SFT parquet
dataset. Its default input is:

```text
data/collected_rollouts/pi05_libero_10/reward
```

Override the rollout location by appending a Hydra argument:

```bash
bash experiments/pi05_wm/train_prefix_input.sh \
  offline_warmup.data_dir=/absolute/path/to/rollout/reward
```

Text switches remain task-owned so extraction and WM construction stay aligned:

```bash
bash experiments/pi05_wm/train_prefix_input.sh \
  task.prefix_input_latent.wm_use_text=false \
  task.prefix_input_latent.policy_use_text=true
```
