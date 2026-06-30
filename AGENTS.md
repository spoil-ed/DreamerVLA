# AGENTS.md

Brief for AI coding agents working on DreamerVLA. For contribution mechanics, commit
style, and PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick orientation:** DreamerVLA is a single-machine VLA + world-model training stack
for LIBERO. Hydra owns configuration. A `Runner` owns one train/eval job. The current
mainline is the OpenVLA-OFT one-trajectory cold-start workflow:

`collect rollouts -> seed replay -> warm up world model + success classifier -> online cotrain`

The mainline experiments are `collect_rollouts_ray` / `collect_rollouts_onetraj`,
`openvla_onetraj_libero_cotrain_noray` (sync pipeline) and
`openvla_onetraj_libero_cotrain_ray` (Ray async manual cotrain), then `eval_libero_vla`.
Everything else is a secondary route or a test fixture — the authoritative mainline vs
secondary split is [spec/06_routes.md](spec/06_routes.md). The command-level reference is
[spec/04_complete_loop.md](spec/04_complete_loop.md). Architecture source documents live
under [spec/](spec/), with [spec/99_manual_notes.md](spec/99_manual_notes.md) as the
highest-priority user guidance. Keep this file as the repository brief, not a history
log. Shipped work and open work live in [docs/HISTORY.md](docs/HISTORY.md) and
[docs/superpowers/TODO.md](docs/superpowers/TODO.md).

---

## Code Structure

- **`dreamervla/`** - main package:
  - `train.py` - Hydra entry. Resolves config, validates it, loads `cfg._target_`,
    then runs `setup -> execute -> teardown`.
  - `config.py` - early validation for logger backends, actor-update routes, batch
    shape, resume paths, sidecar contracts, latent dimensions, Ray resources, and FSDP.
  - `launchers/` - Python launchers. The main pipeline launcher is
    `coldstart_warmup_cotrain.py`.
  - `runners/` - `BaseRunner` plus public runner targets. Current mainline runners are
    `CollectRolloutsRunner`, `ColdStartRayCollectRunner`,
    `OnlineCotrainPipelineRunner`, `OnlineCotrainRunner`, `ManualCotrainRayRunner`,
    and `OnlineCotrainRayRunner`.
    WM, classifier, VLA SFT, and eval runners are also here.
  - `models/` - VLA encoders/policies, world models, actors, critics, reward heads.
  - `algorithms/` - LUMOS/PPO-style update code, actor-update registry, reward-model
    registry, verifier protocol.
  - `dataset/` and `preprocess/` - LIBERO HDF5 datasets, rollout dumps, manifests,
    hidden sidecars, and validation utilities.
  - `envs/` - LIBERO train/eval env wrappers plus `envs/world_model/LatentWorldModelEnv`.
  - `workers/`, `scheduler/`, `hybrid_engines/` - opt-in Ray async cotrain backend:
    env, inference, replay, learner, rollout dump, placement, channels, and weight sync.
  - `diagnostics/` - importable smoke checks and measurement CLIs.
  - `utils/` - checkpoint, logging, metrics, paths, timers, EGL, HF modules, shared helpers.
  - `legacy/` - old artifact utilities. Do not import this from active configs or runners.
- **`configs/`** - Hydra source of truth:
  - `train.yaml` composes `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`,
    `evaluation/`, `logger/`, and `experiment/`.
  - `configs/scripts/coldstart_warmup_cotrain.yaml` defines the launcher-level pipeline.
  - `configs/experiment/` selects complete recipes.
  - `configs/task/` carries LIBERO suite, checkpoint, image/history, and sidecar metadata.
- **`scripts/`** - thin shell launchers. Implementation belongs in `dreamervla/` and
  runs via `python -m`.
- **`tests/`** - `unit_tests/` for contracts and focused behavior; `e2e_tests/` for
  subprocess, Ray, GPU, or real-environment coverage.
- **`data/`** - runtime data root when `DVLA_DATA_ROOT` is not set:
  datasets, checkpoints, collected rollouts, processed data, and outputs.
- **`third_party/`** - vendored LIBERO/OpenVLA/robosuite-style dependencies.

---

## Mainline Flow

Use one of the e2e wrappers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=osmesa

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu render_backend=egl
```

Both scripts set `DVLA_ROOT`, default `DVLA_DATA_ROOT`, activate the `dreamervla`
conda env when available, set `NCCL_NVLS_ENABLE=0`, and call
`python -m dreamervla.launchers.coldstart_warmup_cotrain`.

The launcher composes `configs/scripts/coldstart_warmup_cotrain.yaml`:

- `mode=noray` uses `experiment=collect_rollouts_onetraj`
  (`CollectRolloutsRunner`) for vectorized collection.
- `mode=ray` uses `experiment=collect_rollouts_ray`
  (`ColdStartRayCollectRunner`) for worker-fanout collection.
- Collection always renders with osmesa and writes
  `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/{reward,hidden}` plus
  `collection_manifest.json`.
- The sync cotrain path uses
  `experiment=openvla_onetraj_libero_cotrain_noray`, which composes
  `dreamervla=openvla_onetraj_libero_cotrain_noray`.
- The pipeline runner seeds `OnlineReplay` from collected reward + hidden HDF5 shards,
  warms up the world model and classifier with the same update functions used online,
  then runs online cotrain.
- `cotrain_phase=warmup` writes split warmup checkpoints under
  `${RUN_ROOT}/cotrain/ckpt/`.
- `cotrain_phase=online` resumes those warmup checkpoints and skips collection/warmup.
- `cotrain_engine=async` runs sync warmup first, consolidates a Ray init checkpoint,
  then starts `experiment=openvla_onetraj_libero_cotrain_ray`
  (`ManualCotrainRayRunner`).

Ray async cotrain is explicit and single-node. The target route is manual-notes style:
`LearnerGroup` owns world-model/classifier updates, `ActorGroup` owns VLA FSDP
training, `RolloutGroup` owns no-grad policy inference, and `EnvGroup` owns real/WM
environment interaction.

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

- `resolved_config.yaml`
- `run_manifest.json`
- `checkpoints/`
- `log/tensorboard/`
- `log/wandb/`
- `video/{train,eval}/`
- `diagnostics/`

Keep one invocation under one run root. Pipeline collection and cotrain are separate
sub-roots under `RUN_ROOT`; do not scatter extra artifacts elsewhere.

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
  use `task.openvla_oft.input_tokens.*` and collected HDF5/preprocess metadata. Do not
  copy dimensions by hand between world model, classifier, actor, replay, and sidecars.
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
- Base checkpoints use `${training.out_dir}/checkpoints/global_step_<N>/`.
- Pipeline warmup checkpoints use `${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt` and
  `${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt`.
- Use `BaseRunner.get_global_step_checkpoint_dir` and component checkpoint helpers
  instead of hand-built paths.
- LIBERO Dreamer/OpenVLA evaluation goes through `scripts/eval_libero_vla.sh` and
  `configs/scripts/eval_libero_vla.yaml`.

---

## Extension Points

- **New route:** add a `BaseRunner` subclass, export it from `dreamervla.runners`, add a
  cohesive `configs/<group>/...yaml`, and add an `experiment/<name>.yaml`. Add a shell
  launcher only when `python -m dreamervla.train experiment=<name>` is not enough.
- **New actor update:** register an `ActorUpdateRoute` in `dreamervla/algorithms/registry.py`
  and select it with `algorithm.update_type`.
- **New LUMOS reward model:** implement the reward protocol and register it in
  `dreamervla/algorithms/reward/`.
- **New verifier/classifier:** satisfy `algorithms/verifier/SuccessVerifier` and select
  it through the `classifier` Hydra component.
- **New VLA, actor, WM, critic, or dataset:** implement the existing protocol/kwargs
  contract and wire it through Hydra. Do not add `if model == ...` branches to training
  loops when a registry or target can express the choice.
- **New env:** LIBERO is the stable env/data surface today. Adding another env requires
  task config, rollout record schema, reward labels, and tests.

---

## Optional Components

- Target Ray async cotrain (`ManualCotrainRayRunner`) is the mainline, available through
  `experiment=openvla_onetraj_libero_cotrain_ray`.
- Legacy Ray async cotrain (`OnlineCotrainRayRunner`) survives only as the shared base
  config `dreamervla/openvla_onetraj_libero_cotrain_ray_base` and the Ray smoke fixtures
  (`online_cotrain_ray_synthetic`, `online_cotrain_ray_dreamervla_tiny`); it has no
  standalone mainline experiment.
- RynnVLA (alternative VLA backbone) and the OpenVLA-OFT VLA/WM/classifier stage variants
  remain as secondary routes; the full mainline-vs-secondary list is in
  [spec/06_routes.md](spec/06_routes.md). Do not make them the default path.
- `scheduler/`, `workers/`, and `hybrid_engines/` are backend primitives. Keep them
  behind Hydra-selected runners.

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
