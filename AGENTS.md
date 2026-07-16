# AGENTS.md

Brief for AI coding agents working on DreamerVLA. For contribution mechanics, commit
style, and PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick orientation:** DreamerVLA is a single-machine VLA + world-model training stack
for LIBERO. Hydra owns configuration. A `Runner` owns one train/eval job. The current
mainline is the OpenVLA-OFT one-trajectory cold-start workflow:

`collect rollouts -> seed replay -> warm up world model + success classifier -> online cotrain`

The mainline experiments are `collect_rollouts`, independent WM/classifier warmup,
`openvla_libero`, and `eval_cotrain`. Ray is the implementation backend
for collection and cotrain, so public route names do not carry a `ray` suffix.
The command-level reference is [spec/04_complete_loop.md](spec/04_complete_loop.md).
Architecture source documents live under [spec/](spec/), with
[spec/99_manual_notes.md](spec/99_manual_notes.md) as the highest-priority user
guidance. Keep this file as the repository brief.

The official-data WM and classifier recipes remain supporting capacity checks; they
do not replace the collect/warmup/online-cotrain flow.

---

## Code Structure

- **`dreamervla/`** - main package:
  - `train.py` - Hydra entry. Resolves config, validates it, loads `cfg._target_`,
    then runs `setup -> execute -> teardown`.
  - `config.py` - early validation for logger backends, actor-update routes, batch
    shape, resume paths, sidecar contracts, latent dimensions, Ray resources, and FSDP.
  - `launchers/` - thin Python launchers for one command, generic train/eval dispatch,
    and shell workflow execution.
  - `runners/` - `BaseRunner` plus public runner targets. Current mainline runners are
    `RolloutCollectionRunner`, `CotrainRunner`, `DreamerRunner`, `WorldModelTrainingRunner`,
    `SuccessClassifierTrainingRunner`, and `LIBEROVLAEvaluationRunner`.
    WM, classifier, VLA SFT, and eval runners are also here.
  - `models/` - embodiment model implementations only. `models/embodiment/`
    contains VLA/encoder code and world-model modules; VLA and encoder are the
    same embodiment boundary.
  - `algorithms/` - LUMOS/PPO-style update code, actor modules, critic/classifier
    modules, actor-update registry, reward-model registry, and verifier protocol.
    Critic and classifier are the same value/verifier boundary and live under
    `algorithms/critic/`.
  - `dataset/` and `preprocess/` - LIBERO HDF5 datasets, rollout dumps, manifests,
    hidden sidecars, and validation utilities.
  - `envs/` - `envs/libero/{libero_env.py,utils.py,venv.py}` plus
    `envs/world_model/LatentWorldModelEnv`.
  - `workers/`, `scheduler/`, `hybrid_engines/` - Ray mainline backend:
    env, inference, replay, learner, rollout dump, placement, channels, and weight sync.
  - `diagnostics/` - executable install, eval, smoke, and measurement CLIs.
  - `runtime/` - shared local runner support, including metrics, offline warmup,
    collection adapters, and cotrain evaluation.
  - `utils/` - checkpoint, logging, metrics, paths, timers, EGL, HF modules, shared helpers.
- **`configs/`** - Hydra source of truth:
  - `train.yaml` composes `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`,
    `evaluation/`, `logger/`, and `experiment/`.
  - `configs/experiment/` selects complete recipes.
  - `configs/task/` carries LIBERO suite, checkpoint, image/history, sidecar metadata,
    and the task-owned classifier model, data target, and input contract.
- **`scripts/`** - thin shell launchers. Implementation belongs in `dreamervla/` and
  runs via `python -m`; defaults belong to Hydra, not shell variables.
- **`tests/`** - `unit_tests/` for contracts and focused behavior; `e2e_tests/` for
  subprocess, Ray, GPU, or real-environment coverage.
- **`data/`** - runtime data root when `DVLA_DATA_ROOT` is not set:
  datasets, checkpoints, collected rollouts, processed data, and outputs.
- **`third_party/`** - ignored, read-only upstream runtime dependencies. Inspect
  them when needed, but never edit or stage them from this repository.

---

## Mainline Flow

The retained shell surface exposes train and eval separately:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm-run/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier-run/checkpoints/latest.ckpt

bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/cotrain-run
```

These scripts contain no training defaults. Train selects
`experiment=openvla_libero`; eval selects `eval_cotrain`. Collection is
`experiment=collect_rollouts` and writes reward/hidden shards plus
`collection_manifest.json`. WM and classifier warmup use their independent runners;
their checkpoints are passed to cotrain explicitly.

Cotrain is a single-node Ray implementation with manual-notes-style groups:
`LearnerGroup` owns world-model/classifier updates, `ActorGroup` owns VLA FSDP
training, `RolloutGroup` owns no-grad policy inference, and `EnvGroup` owns real/WM
environment interaction.

`DreamerRunner` is the frozen-WM/CLS imagined-RL specialization of
`CotrainRunner`. It reuses the same Ray loop and complete checkpoint/resume
implementation; it does not own a copied training loop or a second shell launcher.

## 参考实现与学习要求

manual cotrain 实现应以同级 `RLinf` 工作区作为 Group、Worker、
channel data flow 和 training loop 组织方式的参考实现。最接近的路线是 RLinf
embodiment，尤其是 `rlinf/runners/embodied_runner.py`、
`rlinf/workers/env/env_worker.py`、`rlinf/workers/rollout/hf/huggingface_worker.py`
和 `rlinf/data/embodied_io_struct.py`。

修改 DreamerVLA cotrain 内部实现前，必须理解 RLinf 如何拆分 `ActorGroup`、
`RolloutGroup` 和 `EnvGroup`：Actor 负责 FSDP training，Rollout 负责 no-grad
HF/BasePolicy inference 并从 Actor 拉取权重，Env 负责 real/imagined environment
stepping 和 trajectory assembly。DreamerVLA manual route 额外增加 `LearnerGroup`，
用于 world-model/classifier update。

---

## How Training Runs

`python -m dreamervla.train experiment=<name> task=<suite>` does this:

1. Register DreamerVLA OmegaConf resolvers.
2. Force `training.distributed_strategy=ddp` when launched under `torchrun` with
   `WORLD_SIZE > 1`.
3. Resolve Hydra config and call `dreamervla.config.validate_cfg`.
4. Resolve `cfg._target_` to a runner class.
5. Run `BaseRunner.setup()`, `execute()`, and `teardown()`.

`BaseRunner` writes reproducibility artifacts under `${training.out_dir}`:

- `run_manifest.json`
- `checkpoints/`
- `checkpoint_hf/` (only when explicitly enabled)
- `logs/`
- `tensorboard/`
- `wandb/`
- `video/{train,eval}/`
- `diagnostics/`
- `.hydra/`

By default one invocation owns
`${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs}/${run.name}/${run.timestamp}`. Resume reuses
the checkpoint's owning run root; do not create a second timestamp or scatter extra
artifacts elsewhere. Evaluation is the deliberate exception: its run root is
`${run.output_root}/eval/${eval.task_suite_name}` with no timestamp layer.

---

## Configuration Contracts

- **Hydra is the source of truth.** Dims, widths, horizons, batch sizes, checkpoint
  paths, sidecar names, task behavior, logger backends, precision, DDP/FSDP, and Ray
  placement come from config.
- **Validation checks relationships.** It must not choose training behavior. Use
  assertions and `validate_cfg` to prove values align, not to set hidden defaults.
- **Use config-selected construction.** Normal components use Hydra `_target_` and
  `hydra.utils.instantiate`. Ray worker configs use the existing `target` + `kwargs`
  builders. Do not hardcode concrete model/dataset/worker classes inside train loops.
- **Name by role.** Prefer contract names such as `OnlineReplay`, `VecRolloutEnv`,
  `PixelHiddenSequenceDataset`, and `LatentWorldModelEnv`. Do not name core modules after
  one checkpoint or one artifact unless that external boundary is the whole contract.
- **Keep hidden concepts separate.** `wm_obs_dim`, `token_count`, `token_dim`,
  `chunk_size`, and sidecar keys describe external VLA/sidecar data. `model_dim`,
  RSSM/TSSM width, heads, depth, and MLP sizes describe internal world-model capacity.
- **Derive downstream shapes from task + sidecar metadata.** For OpenVLA-OFT routes,
  use `task.openvla_oft.hidden_token.*` and collected HDF5/preprocess metadata. Do not
  copy dimensions by hand between world model, classifier, actor, replay, and sidecars.
  The one-trajectory mainline persists `hidden_token [256,4096]`; the
  decoder's internal action slots are not an observation sidecar.
- **Checkpoint-specific settings follow the checkpoint.** History, image rotation,
  prompt style, proprio/state inclusion, and action-head type are task/checkpoint
  metadata. Do not encode them as fixed schemes in runners.
- **Optional components are opt-in.** Build and validate only what the active config
  declares. Use registries, protocols, and narrow capability checks instead of broad
  "not supported" branches.

---

## Metrics, Checkpoints, Evaluation

- Route metrics through `BaseRunner.log_metrics`.
- Use namespaces: `train/`, `eval/`, `env/`, `rollout/`, `replay_buffer/`,
  `time/`, and `sync/`.
- Logger backends come from `runner.logger.logger_backends`; defaults use TensorBoard
  and W&B where the active experiment declares them.
- Every training route writes the resumable payload to
  `${training.out_dir}/checkpoints/latest.ckpt`.
- Metric-selected files are flat siblings named
  `epoch=<completed-epoch>-<metric>=<value>.ckpt`; the metric name and top-k count
  come from `checkpoint.topk`.
- `checkpoint_hf/` is a sibling of `checkpoints/` and is created only when HF export
  is explicitly enabled. Never create route or step subdirectories below `checkpoints/`.
- Resume accepts a concrete checkpoint, `checkpoints/`, or a run root and resolves
  directories to `latest.ckpt`; historical layouts remain read-only fallbacks.
- Cotrain evaluation goes through `scripts/experiments/cotrain/eval.sh` and
  `configs/experiment/eval_cotrain.yaml`.

---

## Extension Points

- **New route:** add a `BaseRunner` subclass, export it from `dreamervla.runners`, add a
  cohesive `configs/<group>/...yaml`, and add an `experiment/<name>.yaml`. Add a shell
  launcher only when `python -m dreamervla.train experiment=<name>` is not enough.
- **New actor update:** register an `ActorUpdateRoute` in `dreamervla/algorithms/registry.py`
  and select it with `algorithm.update_type`. Actor modules live under
  `dreamervla/algorithms/actor/`.
- **New LUMOS reward model:** implement the reward protocol and register it in
  `dreamervla/algorithms/reward/`.
- **New verifier/classifier/critic:** satisfy `algorithms/verifier/SuccessVerifier`
  when used as a success verifier. Implement critic/classifier modules under
  `dreamervla/algorithms/critic/` and select them through Hydra.
- **New VLA, encoder, WM, actor, critic/classifier, or dataset:** implement the
  existing protocol/kwargs contract and wire it through Hydra. VLA/encoder/WM code
  belongs under `dreamervla/models/embodiment/`; actor code belongs under
  `dreamervla/algorithms/actor/`; critic/classifier code belongs under
  `dreamervla/algorithms/critic/`. Do not add `if model == ...` branches to
  training loops when a registry or target can express the choice.
- **New env:** LIBERO is the stable env/data surface today. Adding another env requires
  task config, rollout record schema, reward labels, and tests.

---

## Backend Components

- `scheduler/`, `workers/`, and `hybrid_engines/` are Ray backend primitives. Keep
  them behind Hydra-selected public runners; do not create a parallel non-Ray route.

---

## When Things Go Wrong

- Run `bash scripts/install/60_verify.sh` before large OpenVLA-OFT jobs. It checks the
  OpenVLA-OFT transformer fork and the pinned package set used by the tutorial.
- OpenVLA-OFT currently expects `peft==0.11.0`; newer `peft` versions can import symbols
  missing from the transformer fork.
- Select rollout rendering with the launcher/config knob `render_backend=egl|osmesa`.
  Collection stays on osmesa.
- For EGL, Ray async placement owns `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`
  per EnvWorker. Do not bypass the placement contract with ad-hoc env vars.
- NCCL/CUDA timeouts under DDP usually mean a rank diverged first. Read the rank-0 log
  and keep the DDP sync guards in `dreamervla/algorithms/ppo/outcome.py`.

---

## Style

- Python 3.11.
- Type hints and docstrings on public APIs.
- Static YAML only. Derive runtime values in runners or builders, then validate.
- Shell scripts are one-command launchers. No loops, `case`, functions, or custom arg
  parsers; use Python/Hydra for iteration and dispatch. `if` is acceptable for
  run/skip/required-input guards.
- No bare `print` in training-loop code except concise rank-0 progress lines already
  used by runners. Prefer runner logging and `utils/json_logger.py`.
- New behavior needs tests under `tests/`; GPU/Ray/real-env coverage belongs in
  `tests/e2e_tests/` and must be gated appropriately.
- Commits use Conventional Commits, about 72 characters, imperative mood, and
  `git commit -s`.

---

## Further Reading

- [Architecture overview](spec/00_overview.md)
- [Project goals](spec/01_goal.md)
- [Naming principles](spec/02_naming.md)
- [Coding style](spec/03_coding_style.md)
- [Complete cotrain loop](spec/04_complete_loop.md)
- [Manual notes](spec/99_manual_notes.md)
- [Parameter reference](docs/PARAMETERS.md)
- [Install](docs/install.md)
- [Data layout](docs/data_layout.md)
- [Config registry](configs/README.md)
- [Script registry](scripts/README.md)
- [Repository structure](docs/repository_structure.md)
- [Docs index](docs/README.md)
- [README](README.md) / [中文 README](README.zh-CN.md)
