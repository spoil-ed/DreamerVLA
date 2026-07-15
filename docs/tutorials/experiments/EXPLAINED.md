# Experiment tutorials — background & rationale

The tutorial files in this directory are **step-only recipes** (commands, no prose).
All explanation lives here, once per topic. Parameter reference:
[`docs/PARAMETERS.md`](../../PARAMETERS.md).

## How the recipes are structured

Each recipe is a clean-checkout → install → download → preprocess → train → eval
sequence. Normal changes are Hydra overrides:

```bash
gpus=0,1 ngpu=2 batch_size=16 num_workers=4 num_epochs=20 out_dir=/tmp/run
```

Shell launchers are thin: they set project/data roots and call Hydra-backed Python
entry points. The release tree keeps current wrappers under `scripts/`, such
as `train_dreamervla.sh` and the cold-start e2e launchers. All experiment choice is
`experiment=<name>` plus ordinary Hydra keys. The pipeline task name is the Hydra
`task=` value (snake_case, e.g. `task=openvla_onetraj_libero`). On-disk data artifacts
use `task.artifact_name` (e.g.
`OpenVLA_Onetraj_LIBERO_libero_goal`), so paths inside commands intentionally mix the
snake_case `task=` token with the CamelCase artifact directory.

## Recipe map

| Recipe | Scheme | What it builds |
| --- | --- | --- |
| `openvla_onetraj_libero` | discrete-token route | OpenVLA-OFT one-trajectory baseline |
| `openvla_onetraj_libero` hidden-token WM | query_before | OFT projected vision-token WM (offline + online cotrain) |
| `openvla_onetraj_coldstart_libero` | cold-start | fresh rollout collection (+ warmup + cotrain) |

## Observation contract

The OpenVLA-OFT mainline persists projected current-frame vision embeddings before
the language-model action positions. With one image, the sidecar is `[T,256,4096]`
and declares `obs_hidden_source=hidden_token`. The actor bridges those 256
source tokens to the discrete action decoder; its internal action slots are not a WM
observation and are never written as a sidecar.
- **Discrete-token route:** the OpenVLA-OFT one action-probability route used by the
  one-trajectory baseline. L1/action-query checkpoints are explicitly rejected.

## World model architecture (WM chunk predictor)

The world model is the WM paradigm migrated onto **discrete** OpenVLA-OFT latents
(no L1 head). The action is encoded and concatenated to every obs token channel;
`predict_next_chunk` rolls K env-steps autoregressively. Training uses `num_hist=3`
autoregressive recursion (a 3-term window, free-running).

The hidden-token route uses `token_count=256`, `token_dim=4096`, and
`wm_obs_dim=1048576`. Proprio, language, and action conditioning widths are separate
from this external observation shape and are derived from Hydra metadata.
The rollout length is a hyperparameter: with horizon H, chunk K, N chunks,
`sequence_length = H + N*K + 1`. `num_hist=3` is required by the recursion. Predictor
profile values live in the `worldmodel/` configs.

> The mainline `model_dim` is **4148**. This is adjustable but must match the WM
> checkpoint being loaded.

## WM single-episode overfit probe

`wm_single_episode_overfit.py` is a diagnostic script, not a training launcher.  It is
dry-run by default and only trains when `--run` is supplied.  The probe fixes one
LIBERO episode, trains the Chunk-WM on sliding windows from that episode, and records
imagined rollouts under three action interventions:

- `true`: demo action chunk
- `zero`: all-zero action chunk
- `random`: random action chunk with a deterministic seed

The important comparison is whether `true` action rollout becomes clearly better than
`zero` / `random` in `*_mse`, `*_cos`, and classifier max score.  If all three stay
nearly identical, the WM is fitting mostly from latent history instead of using the
action channel strongly enough.

## Memory / OOM (online cotrain)

The LUMOS RL update imagines an **effective batch** through the world model at once:

```
B_eff = dataloader.batch_size × algorithm.imag_last × algorithm.ppo_rollouts_per_start
```

`B_eff` (not the raw batch) is the memory dial. Fast fixes:

1. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. a sane `algorithm.imag_last` (start states per replay window; default 4, diverse
   strided selection)
3. cap `algorithm.lumos.episode_max_steps` / `algorithm.ppo_rollouts_per_start`

With `batch_size=12, imag_last=4` the hidden-token route fits an 80GB GPU at ~66.7 GB.
The OFT latent is large; shrink `B_eff` first (chunk-granular video, sliced `lm_head`).
`training.debug=true` (or `debug=true` to the e2e scripts) runs a tiny smoke instead of
the full pipeline.

## Logging

Grouped training defaults to TensorBoard + W&B online.

```bash
logger=tensorboard_wandb runner.logger.wandb_mode=online   # default
logger=tensorboard_wandb runner.logger.wandb_mode=offline  # offline W&B
logger=tensorboard                                          # local TensorBoard only
logger=wandb runner.logger.wandb_mode=online               # W&B online only
```

TensorBoard events: `${training.out_dir}/tensorboard`; W&B run files:
`${training.out_dir}/wandb`.

```bash
tensorboard --logdir "${OUT_DIR}/tensorboard" --host 0.0.0.0 --port 6006
ssh -L 6006:localhost:6006 user@host        # remote: forward then open localhost:6006
wandb sync "${OUT_DIR}/wandb/wandb/offline-run-<FIRST>-<ID>"
# After checkpoint resume, append each later offline segment to the same web run:
wandb sync --id "$(cat "${OUT_DIR}/wandb/run_id.txt")" --append \
  "${OUT_DIR}/wandb/wandb/offline-run-<LATER>-<ID>"
```

"TensorFlow logging" in these recipes means the TensorBoard event writer, not a
separate TensorFlow backend.

## Classifier corpora

Classifier and LUMOS training need both **success and failure** rollout corpora. The
standard LIBERO download gives success demos. Without failure demos + matching
sidecars, stop after WM training or use the actor-critic route where applicable.

## Cold-start flow

Cold-start collects fresh OpenVLA-OFT one-trajectory rollouts (reward HDF5 + action-
hidden sidecar) under a marked `collected_rollouts/` root (never the offline
`processed_data/`), then warms up the WM + classifier and runs RL cotrain. The
`task.openvla_oft.*` block is the source of truth for dims, ckpts, and suite binding —
one VLA ckpt per LIBERO suite, explicit `task=` selection, no silent defaulting.

## Rendering on this host

LIBERO rendering: EGL crashes in robosuite `read_pixels` on some hosts; use
`export MUJOCO_GL=osmesa` (and `PYOPENGL_PLATFORM=osmesa`). Multi-GPU DDP cotrain may
need `export NCCL_NVLS_ENABLE=0` to avoid NVLink-SHARP collective hangs.

## OpenVLA-OFT needs the transformers fork

The OFT route requires the moojink transformers fork
(`github.com/moojink/transformers-openvla-oft`, bidirectional Llama attention) — same
`4.40.1` version string as vanilla, but vanilla yields garbage actions (≈0% success).
The install scripts make this fork **the single authoritative transformers in the main
`dreamervla` env** (`scripts/install/40_third_party.sh`); `scripts/install/60_verify.sh`
FATAL-checks that the installed transformers is the fork, so no separate env is needed.
The OpenVLA-OFT / RLinf action contract is encoded in the current rollout env,
worker, and config tests; keep those tests as the source of truth when changing
the route.

## Validation notes

- [`spec/`](../../../spec/00_overview.md) — architecture entry points for overview,
  complete loop, Ray implementation, current implementation, and data contracts.
