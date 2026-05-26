# CLAUDE.md

DreamerVLA — frozen pi0 VLA encoder + DreamerV3 RSSM world model + actor-critic on LIBERO.

## 1. Data flow

```
data/libero/           →   src/preprocess/*   →   data/processed_data/<variant>/   →   dataloader
(raw HDF5, input only)                            (sidecars consumed downstream)
```

Downstream stages never read `data/libero/` directly. Sidecar naming: `<suite>_<filter>_t_<timesteps>_<backbone>_<hidden_kind>_<prompt_style>_h<history>`.

## 2. Workflow

Three layers, strictly separated:

- **`scripts/*.sh`** — thin shell wrappers (env vars + GPU defaults), one per experiment.
- **`src/cli/train.py` / `eval.py`** — model-agnostic entry points. Load a Hydra config and dispatch to the workspace named by `_target_`. Lifecycle: `setup() → execute() → teardown()`.
- **`src/workspace/*.py`** — orchestrates a run by composing prebuilt blocks (`dataloader`, `env`, `model`, `algorithm`, `trainer`, `logger`). Wires components; does not implement them.

Dispatch uses Hydra `_target_` (single underscore). Example:

```yaml
_target_: src.workspace.JointDreamerVLAWorkspace
dataset:   { _target_: src.dataloader.libero_pixel_rynn_hidden_sequence_dataset.LIBEROPixelRynnHiddenSequenceDataset, ... }
model:     { _target_: src.models.world_model.dreamerv3_pixel_rynn_backbone_wm.DreamerV3PixelRynnBackboneWorldModel, ... }
```

New route = new workspace + new config + new shell wrapper. Don't bypass.

### Shell wrappers are dumb on purpose

The three canonical shells at `scripts/` root (`train_wm.sh`, `train_vla.sh`,
`train_dreamervla.sh`) must stay ≤ ~35 lines. Their only job:

1. `cd` to repo root, set `PYTHON` / `NGPU` / `CONFIG` / `MASTER_PORT`
2. Branch single-GPU vs `torchrun`
3. `exec python -m src.cli.train --config-name "$CONFIG" "$@"`

**Things that do NOT belong in the shell** (history lesson — `train_wm.sh`
once bloated to 466 lines doing all of these):

- Default hyperparameters (`BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, `WARMUP`,
  `GRAD_CLIP`, etc.). These live in the YAML.
- Per-kind `KIND_OVERRIDES` that re-set YAML defaults via env vars.
  Same value in two places = bug magnet.
- DDP toggles (`training.distributed_strategy=ddp`, `data_parallel=false`).
  `src/cli/train.py::_auto_apply_distributed` reads `RANK` / `WORLD_SIZE` from
  the env and applies them automatically when launched under `torchrun`.
- Output-dir construction. Each YAML's `training.out_dir` uses
  `${oc.env:OUT_DIR,data/outputs/<bucket>/<arch>/${now:%Y%m%d_%H%M%S}}` — set
  `OUT_DIR=...` to override, otherwise a timestamped default is generated.
- Legacy env-var alias chains (`RYNN_PIXEL_DDP / RYNN_BACKBONE_DDP /
  ACTION_HIDDEN_DDP / DDP`). Pick one name and stick to it; rename via
  `git mv`, do not fallback-chain.
- Smoke flags. Just pass the overrides on the CLI:
  `bash scripts/train_wm.sh training.max_steps=1 dataloader.num_workers=0 viz.enabled=false`.

If a value needs to change per-run, it goes on the Hydra CLI (`task=libero_object`,
`dataloader.batch_size=8`). If it needs to change per-config-route, it goes in
the YAML. The shell only sees `CONFIG`, `NGPU`, `OUT_DIR`, and pass-through `$@`.

## 3. File layout

| Path | Role |
|---|---|
| `src/` | All model + pipeline implementations (`workspace/`, `dataloader/`, `models/`, `algorithms/`, `env/`, `trainer/`, `preprocess/`, `utils/`, `cli/`). |
| `scripts/` | Experiment launchers (bash) + standalone diagnostic/measurement Python. Glue only. |
| `configs/` | Hydra configs, one per run. `_target_` picks the workspace and nested components. |
| `data/` | All inputs and outputs (raw, processed, ckpts, outputs, logs). Gitignored. |
| `docs/` | Writeups, paper drafts, history, TODOs. |
| `dependencies/` | Source-built / vendored deps used by the install script. |

## 4. `data/outputs/` structure

```
data/outputs/
├── dreamervla/   ├── vla/   ├── worldmodel/   ├── eval/   ├── logs/   └── README.md
```

Per-bucket convention: `<architecture>/<implementation_variant>/<param_or_experiment_tag>/`. Each run folder has `ckpt/`, run log, resolved Hydra config, and TB/wandb traces. `data/outputs/README.md` is the authoritative index of retained reference runs.

---

## Environment

- Repo at `/mnt/data/spoil/workspace/DreamerVLA` (migrated from `/home/user01/liops/...` on 2026-05-25; older paths may linger).
- Conda env `dreamervla` — `/home/user01/miniconda3/envs/dreamervla/bin/python` (Python 3.11, torch 2.5.1+cu124).
- `/home` is ~99% full — new artifacts under `/mnt/data/spoil/`. `/dev/shm` (1 TB) for hot scratch.
- Multi-GPU H100/H800/A100 with parallel `tmux` sessions — check `nvidia-smi` and `tmux ls` first.
- Pinned: `transformers==4.40.1`, `numpy==1.26.4`, `torch==2.5.1`, `xformers==v0.0.28.post3`. LIBERO needs `MUJOCO_GL=osmesa`.

## Common commands

```bash
python -m src.cli.train --config-name <config> [overrides ...]    # universal entry
bash scripts/<launcher>.sh                                        # wraps the above
bash scripts/eval_libero_vla.sh --ckpt_path <ckpt> --task_suite libero_goal --num_episodes 10
python -m pytest tests/<file>.py -xvs
```

Common env vars: `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `BATCH_SIZE`, `RUN_TAG`, `DETACH=1`, `DRY_RUN=1`, `WM_SMOKE=1`.

## Doc index

`README.md` (setup), `docs/TODO.md`, `progress.md` / `findings.md` (running notes), `scripts/README.md`, `configs/README.md`, `data/outputs/README.md`.
