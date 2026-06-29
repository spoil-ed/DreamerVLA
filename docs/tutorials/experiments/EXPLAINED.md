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

Shell launchers (`scripts/train_vla.sh`, `train_wm.sh`, `train_dreamervla.sh`) are
thin: they set project/data roots and call the grouped Hydra entry. All experiment
choice is `experiment=<name>` plus ordinary Hydra keys. The pipeline task name is the
Hydra `task=` value (snake_case, e.g. `task=openvla_onetraj_libero`). On-disk data
artifacts keep their historical names via `task.artifact_name` (e.g.
`OpenVLA_Onetraj_LIBERO_libero_goal`), so paths inside commands intentionally mix the
snake_case `task=` token with the CamelCase artifact directory.

## Recipe map

| Recipe | Scheme | What it builds |
| --- | --- | --- |
| `rynnvla_libero` | Scheme A (legacy action-hidden) | RynnVLA + LIBERO-Goal full pipeline |
| `openvla_onetraj_libero` | discrete-token route | OpenVLA-OFT one-trajectory baseline |
| `openvla_onetraj_libero` action-hidden WM | Scheme A (query_after) | OFT action-slot hidden WM (offline + online cotrain) |
| `openvla_onetraj_libero` backbone-latent WM | Scheme 1 (query_before) | OFT input-token (backbone) latent WM |
| `openvla_onetraj_coldstart_libero` | cold-start | fresh rollout collection (+ warmup + cotrain) |

## Latent schemes

- **Scheme A (action-hidden, query_after):** action-slot hidden tokens produced by the
  VLA action head are consumed directly by the WM, classifier, and DreamerVLA actor.
  Mainline route. Short sequence (action-hidden, ~56 tokens).
- **Scheme 1 (backbone-latent, query_before):** the WM target is the future
  backbone/DINO-style visual-language latent (input tokens, ~512 tokens, ~9× longer).
  The actor is `LatentToOpenVLADiscreteTokenActor` (a discrete bridge over the latent).
- **Discrete-token route:** the OpenVLA-OFT one action-probability route used by the
  one-trajectory baseline (discrete tokens, not the L1 Gaussian route — the L1 route is
  a separate, not-yet-validated gap).

## World model architecture (DINO-WM chunk predictor)

The world model is the DINO-WM paradigm migrated onto **discrete** OpenVLA-OFT latents
(no L1 head). The action is encoded and concatenated to every obs token channel;
`predict_next_chunk` rolls K env-steps autoregressively. Training uses `num_hist=3`
autoregressive recursion (a 3-term window, free-running).

Sized per scheme under a ~1B cap:

- **query_after** (action-hidden, seq≈168): full-width attention inner =
  `heads*dim_head = 4096 = model_dim` (no compression of the dense 4096-d VLA tokens),
  lean FFN `mlp_dim = model_dim`. ~610M. `model_dim 4106`.
- **query_before** (input-token, seq≈1536): half-width attention inner = `2048 =
  0.5*model_dim`, lean FFN. ~313M, because its sequence is ~9× longer. `model_dim 4106`.
- RynnVLA action-hidden WM: smaller latent, `model_dim 1034`.

The rollout length is a hyperparameter: with horizon H, chunk K, N chunks,
`sequence_length = H + N*K + 1`. `num_hist=3` is required by the recursion. Predictor
profile values live in the `worldmodel/` configs.

> The mainline `model_dim` is **4106** (8 configs) and the RynnVLA action-hidden WM is
> **1034**. These are adjustable but must match the WM checkpoint they load — a run that
> resumes a `model_dim=1024` warmup checkpoint is incompatible with the current sizing.

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

With `batch_size=12, imag_last=4` the action-hidden route fits an 80GB GPU at ~66.7 GB.
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

TensorBoard events: `${training.out_dir}/log/tensorboard`; W&B run files:
`${training.out_dir}/log/wandb`.

```bash
tensorboard --logdir "${OUT_DIR}/log/tensorboard" --host 0.0.0.0 --port 6006
ssh -L 6006:localhost:6006 user@host        # remote: forward then open localhost:6006
wandb sync "${OUT_DIR}/log/wandb"           # upload an offline run
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
See
[`RLinf_aligned_LIBERO_rollout_execution_plan.md`](../../archive/plans/RLinf_aligned_LIBERO_rollout_execution_plan.md)
for the full OpenVLA-OFT / RLinf action contract and root-cause record.

## Validation notes

- [`RLinf_aligned_LIBERO_rollout_execution_plan.md`](../../archive/plans/RLinf_aligned_LIBERO_rollout_execution_plan.md)
  — the OpenVLA-OFT / RLinf action contract and the shared standalone / no-Ray / Ray
  rollout core.
- [`spec/`](../../../spec/00_overview.md) — architecture entry points for overview,
  complete loop, Ray implementation, current implementation, and data contracts.
