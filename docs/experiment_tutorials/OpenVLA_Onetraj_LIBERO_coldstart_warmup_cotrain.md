# OpenVLA-OFT Cold-Start Collection And Warmup

This recipe reproduces the OpenVLA-OFT one-trajectory LIBERO cold-start flow:

```text
collect rollouts -> reward HDF5 + obs_embedding sidecar -> offline WM/classifier warmup -> online cotrain
```

The default now runs the **full** pipeline end-to-end: collect → offline WM/classifier
warmup (`2000`/`2000` steps) → **online cotrain** (`online_rollout.total_env_steps=200000`).
Add `debug=true` to run the *same* pipeline at a tiny `debug_*` scale (a fast,
memory-fittable smoke). Warmup-only is still available with
`warmup.total_env_steps=0` (it does not load the VLA).

## Requirements

Activate the project environment and point the data root at the prepared assets:

```bash
cd DreamerVLA
conda activate dreamervla
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# Multi-GPU DDP cotrain: disable NCCL NVLS (NVLink SHARP) to avoid collective
# hangs/init failures on some driver+topology combos.
export NCCL_NVLS_ENABLE=0
```

Required files for the selected suite:

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<suite-ckpt>/
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<suite-ckpt>/dataset_statistics.json
${DVLA_DATA_ROOT}/datasets/libero/<suite>/*.hdf5
```

The OpenVLA-OFT checkpoint must use the OpenVLA-OFT transformers fork. Run the
install verifier before long jobs:

```bash
bash scripts/install/60_verify.sh
```

## Tasks

| Launcher `task=` | Hydra task | LIBERO suite |
| --- | --- | --- |
| `goal` | `OpenVLA_Onetraj_ColdStart_LIBERO` | `libero_goal` |
| `object` | `OpenVLA_Onetraj_ColdStart_LIBERO_Object` | `libero_object` |
| `spatial` | `OpenVLA_Onetraj_ColdStart_LIBERO_Spatial` | `libero_spatial` |
| `10` | `OpenVLA_Onetraj_ColdStart_LIBERO_10` | `libero_10` |

All model, world-model, actor, classifier, token, and action dimensions are
derived from `task.openvla_oft`.

## World Model (DINO-WM chunk predictor)

The world model is the DINO-WM paradigm migrated onto **discrete** OpenVLA-OFT
latents — no L1 action head anywhere on this route. The VLA backbone token
dimension drives the predictor's residual width.

### Latent and conditioning

- Latent stage `query_after`: the OFT **action-query hidden** sidecar, shape
  `(time_horizon, 229376)` → `token_count=56`, `token_dim=4096`.
- DINO-WM concat conditioning: the action is encoded to `action_emb_dim=10`,
  tiled to every observation token, and concatenated on the channel axis. The
  residual width is therefore pinned, not free:
  `model_dim = token_dim + action_emb_dim * num_action_repeat = 4096 + 10*1 = 4106`.

### Autoregressive training recursion

`F_ω` consumes a fixed window of `num_hist = 3` latents plus one action and
predicts the next latent. Predicted latents slide back into the window, so the
rollout is free-running (no teacher forcing inside the chunk):

```text
ê_{t+1} = F_w(e_{t-2}, e_{t-1}, e_t,      a_t)
ê_{t+2} = F_w(e_{t-1}, e_t,     ê_{t+1},  a_{t+1})
ê_{t+3} = F_w(e_t,     ê_{t+1}, ê_{t+2},  a_{t+2})
ê_{t+4} = F_w(ê_{t+1}, ê_{t+2}, ê_{t+3},  a_{t+3})
```

By step 4 all three inputs are model predictions; actions are always the real
demo actions. Implemented in `predict_next` / `predict_next_chunk`
(`dreamervla/models/world_model/dino_wm_chunk.py`).

### The rollout length is a hyperparameter

The window depth and the rollout horizon are explicit Hydra knobs, not constants:

| Knob | Symbol | Value | Meaning |
| --- | --- | --- | --- |
| `world_model.num_hist` | H | 3 | history window depth (the 3 inputs to F_w) |
| `world_model.chunk_size` | K | 8 | autoregressive steps per chunk (chunk 0) |
| `world_model.chunk_rollout_chunks` | N | 4 | extra closed-loop chunks (anti-drift) |
| derived horizon | N*K | 32 | total supervised autoregressive steps |
| `dataset.sequence_length` | H + N*K + 1 | 36 | window the dataloader must serve |

`num_hist=3` is required by the recursion above (exactly three history terms).
Changing the horizon (K, N) means matching `dataset.sequence_length = H + N*K + 1`.

### Predictor sizing — efficiency vs capacity

`model_dim=4106` is fixed by the concat rule; the predictor's internal width is
chosen to balance accuracy against compute. The action-hidden sequence is short
(`num_hist * token_count = 3 * 56 = 168` tokens), so full-width attention is
nearly free on this route — capacity is spent where it is cheap.

| profile | inner = heads*dim_head | mlp_dim | depth | total params |
| --- | --- | --- | --- | --- |
| compact (old) | 256 (0.06x) | 1024 | 4 | 55M |
| dino-wm default | 1024 (0.25x) | 2048 | 6 | 207M |
| **balanced (this route)** | **4096 (1.00x)** | **4096** | 6 | **610M** |
| full-width | 4096 (1.00x) | 8192 | 6 | 812M |

This route uses **full-width attention** (`heads=16, dim_head=256` → inner=4096 =
`model_dim`, no compression of the dense 4096-d VLA tokens) with a **lean FFN**
(`mlp_dim=4096`). ~610M params, under the 1B ceiling. All values live in
`configs/dreamervla/online_cotrain_pipeline_openvla_oft_action_hidden.yaml`.

> Memory: ~610M is ~11x the old 55M compact WM. The default full pipeline runs online
> cotrain with the VLA + WM co-resident (memory-heavy — see the **"Memory (online
> cotrain)"** note under **Run** for the `B_eff` breakdown; the short version is
> `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` plus a sane
> `algorithm.imag_last`, or `debug=true`). Warmup-only
> (`online_rollout.total_env_steps=0`) does not load the VLA, so it fits easily. The
> new architecture is incompatible with old `model_dim=1024` warmup checkpoints —
> start fresh.

## Run

No-Ray collector:

```bash
DVLA_ROOT=/path/to/DreamerVLA DVLA_DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal
```

Ray collector:

```bash
DVLA_ROOT=/path/to/DreamerVLA DVLA_DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal
```

Both commands run the **full** pipeline by default. For a fast, low-memory smoke of
the same flow, add `debug=true`:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal debug=true
```

> **Memory (online cotrain) — the WMPO RL update.** The actor update
> (`dino_wmpo_outcome_step`) imagines an **effective batch**
>
> ```
> B_eff = dataloader.batch_size × algorithm.imag_last × algorithm.ppo_rollouts_per_start
> ```
>
> of trajectories through the 610M world model **at once** (one imagined episode
> per start state × GRPO group). Everything heavy scales with `B_eff`: the WM
> attention forward (`to_qkv`), the imagined latent "video" the success classifier
> scores, and the actor log-prob pass. With `batch_size=12, imag_last=4, group=4`
> → `B_eff=192`, which fits an 80GB GPU at **~66.7 GB once you set**:
>
> ```bash
> export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reclaims the warmup-fragmented floor — single biggest win
> ```
>
> The route is already memory-tuned: it (a) **caps imagination starts with
> `algorithm.imag_last`** (default 4, diverse *strided* selection across the
> replay window) instead of imagining from every one of the 36 frames, (b) stores
> the imagined video at the classifier's **chunk granularity** (1/K the frames),
> and (c) computes **only the action-token `lm_head` columns** (not the full
> 32000-vocab logits). So at `batch_size=12` no change is needed beyond
> `expandable_segments`. If you still OOM (e.g. the release profile's
> `batch_size=96` → `B_eff=1536`), shrink `B_eff` in this order: lower
> `algorithm.imag_last` → `dataloader.batch_size` (PER-GPU under DDP) →
> `algorithm.ppo_rollouts_per_start` (keep ≥2 for GRPO variance) →
> `algorithm.wmpo.episode_max_steps` (episode length → video size). `debug=true`
> remains the verified tiny baseline. Run in the `dreamervla` conda env.

Default is the **full** pipeline. Every value is a configurable Hydra override;
`debug=true` swaps the whole right column in automatically (`training.debug=true` →
`OnlineCotrainPipelineRunner._apply_debug_overrides`):

| Parameter (launcher key → Hydra key) | Full (default) | `debug=true` |
| --- | --- | --- |
| `collect.task_ids` | `all` | `all` |
| `collect.episodes_per_task` | `4` | `4` |
| `collect.episode_horizon` | `300` | `300` |
| `warmup.wm_steps` → `training.wm_warmup_steps` | `2000` | `2` |
| `warmup.classifier_steps` → `training.classifier_warmup_steps` | `2000` | `2` |
| `warmup.total_env_steps` → `online_rollout.total_env_steps` | `200000` | `160` |
| `warmup.batch_size` → `dataloader.batch_size` | `96` release / `12` multi_gpu | `2` |
| `online_rollout.episode_horizon` | `200` | `50` |
| `online_rollout.max_train_updates` | unset (run to completion) | `4` |
| `algorithm.imagination_horizon` | `5` | `3` |
| `algorithm.ppo_rollouts_per_start` | `4` | `2` |
| `algorithm.imag_last` (imagination starts/window — B_eff cap) | `4` | `4` (not swapped) |
| `algorithm.wmpo.episode_max_steps` | `300` | `150` |

Adjust them with normal launcher overrides:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
    task=goal \
    collect.envs_per_gpu=16 \
    collect.episodes_per_task=2 \
    warmup.wm_steps=16 \
    warmup.classifier_steps=16
```

Print the exact commands without running them:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal dry_run=true
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal dry_run=true
```

## Output

Each run writes one root:

```text
<run_root>/
  collect/
  coldstart/
    reward/*.hdf5
    hidden/*.hdf5
    hidden/preprocess_config.json
  cotrain/
    ckpt/wm_warmup.ckpt
    ckpt/classifier_warmup.ckpt
```

Use `run_root=...` for a stable output path. Use `skip_collect=true` to reuse an
existing `<run_root>/coldstart/{reward,hidden}` pair and run warmup only.

### Data & checkpoint format conventions

> **Roadmap — planned, not yet implemented.** These are the target conventions;
> current behavior is stated so the doc stays honest.

**Data sharding — slice into reasonable shards.** Cold-start data must be split
into reasonably-sized shards rather than one giant file. `offline_seed` already
reads *every* `*.hdf5` under `reward/` and pairs the same-named shard under
`hidden/`, so multi-shard is fully supported today.
- *Current:* the Ray collector writes one shard per job (`ray_shard_000.hdf5`);
  `scripts/collect_parallel.sh` merges per-GPU jobs as `shard_g{i}.hdf5`.
- *Planned:* shard rotation inside the dump worker — roll a new shard every N
  episodes (or at a size cap) so a single shard never grows unbounded.

**Checkpoints — dual HF + torch, interchangeable.** Everything weight-related
should save/load in **both** formats:
- **HF style** via `save_pretrained` wrapping — each component (world model,
  policy, critic, classifier) written as a HF-style directory (`config.json` +
  `model.safetensors`), so checkpoints are portable and loadable with HF tooling.
- **torch style** — the existing `.ckpt` (`torch.save` of `state_dict`s), kept
  for resume compatibility.
- *Current:* torch `.ckpt` only (`cotrain/ckpt/{wm_warmup,classifier_warmup}.ckpt`,
  `cotrain/ckpt/latest.ckpt`).
- *Planned:* add the HF (`save_pretrained`) path plus a format flag so both are
  produced and consumed interchangeably.

## Inspect Results

Success count:

```bash
python - <<'PY'
from pathlib import Path
import h5py

reward_dir = Path("<run_root>/coldstart/reward")
total = success = 0
for path in sorted(reward_dir.glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        for key in handle["data"]:
            rewards = handle["data"][key]["sparse_rewards"][()]
            total += 1
            success += int(rewards.max() > 0)
print(f"success={success}/{total}")
PY
```

Sidecar shape:

```bash
python - <<'PY'
from pathlib import Path
import h5py

hidden = next(Path("<run_root>/coldstart/hidden").glob("*.hdf5"))
with h5py.File(hidden, "r") as handle:
    ds = handle["data"]["demo_0"]["obs_embedding"]
    print(ds.shape, ds.dtype)
PY
```

For OpenVLA-OFT action-query hidden, expect `(T, 229376)` and `float16`.

Warmup checkpoints:

```bash
python - <<'PY'
from pathlib import Path
import torch

ckpt = Path("<run_root>/cotrain/ckpt")
for name in ("wm_warmup.ckpt", "classifier_warmup.ckpt"):
    payload = torch.load(ckpt / name, map_location="cpu", weights_only=False)
    print(name, sorted(payload))
PY
```

## Manual Commands

No-Ray collect:

```bash
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_manual"
RW="${RUN_ROOT}/coldstart/reward"
HID="${RUN_ROOT}/coldstart/hidden"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.envs_per_gpu=32 \
  collect.memory_fraction=0.9 \
  collect.gpu_id=0 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

Warmup-only:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  offline_warmup.data_dir="${RW}" \
  offline_warmup.hidden_dir="${HID}" \
  offline_warmup.task_id=null \
  'env.task_ids=[0,1,2,3,4,5,6,7,8,9]' \
  training.out_dir="${RUN_ROOT}/cotrain" \
  training.wm_warmup_steps=256 \
  training.classifier_warmup_steps=256 \
  dataloader.batch_size=96 \
  training.classifier_batch_size=512 \
  online_rollout.buffer_size=10000 \
  online_rollout.total_env_steps=0
```

Ray collect:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.memory_fraction=0.9 \
  env.num_workers=16 \
  rollout.max_steps=1200 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

## Validation Commands

Fast Ray contract test:

```bash
python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q
```

Real OFT Ray smoke is gated because it loads the checkpoint and LIBERO:

```bash
DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -q -s
```

## Troubleshooting

- If collection produces no successful episodes, first confirm the route is
  using OpenVLA-OFT action chunks. The collector and Ray inference worker execute
  `task.openvla_oft.chunk_size` actions open-loop before consuming a new chunk.
- If warmup says replay is empty, collect episodes with
  `collect.episode_horizon >= online_rollout.sequence_length`.
- If sidecar validation fails, use one Hydra task consistently for collect and
  warmup; do not mix VLA checkpoint and LIBERO suite manually.
- If Ray hangs during startup, run the synthetic Ray test above and then retry
  with fewer `env.num_workers`.
