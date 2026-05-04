# Config Status

This directory keeps experiment configs, but only a small subset should be used
for the current Dreamer-token path.

## Current Version

Use this for the next training run:

- `dreamer_vla_libero_10_dreamerv3_token_actor_epoch6.yaml`

Meaning:

- DreamerVLA actor/critic training.
- Uses the pretrained DreamerV3 token world model.
- `run_wm_phase: false`, so the world model is loaded and kept as the pretrained
  latent/imagination model.
- `run_actor_critic_phase: true`.
- `num_epochs: 6`.
- Scalar reward head: `reward_bins: 1`.
- Output root: `data/outputs/dreamervla`.

Launch with:

```bash
PYTHON=/home/user01/miniconda3/envs/dreamervla/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
CONFIG_NAME=dreamer_vla_libero_10_dreamerv3_token_actor_epoch6 \
RUN_TAG=token_actor_epoch6 \
bash scripts/train_dreamer_vla.sh
```

## Required / Reproducible Inputs

- `dreamerv3_token_libero_10.yaml`
  - Reproduces the pretrained token world model used by the current DreamerVLA
    actor config.
  - Current checkpoint:
    `data/outputs/worldmodel/dreamerv3_token/dreamerv3_token_libero_10_20260504_075915/ckpt/latest.ckpt`

- `pretokenize_vla_libero_10.yaml`
  - Reproduces the standalone VLA checkpoint.
  - Kept because `epoch=005` is used as an encoder/action-head init in older
    DreamerVLA configs, and `epoch=006` is the checkpoint evaluated at 88.8%
    on LIBERO-10.

- `eval_libero_vla.yaml`
  - Formal LIBERO evaluation config.
  - Default checkpoint points to:
    `data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_10_20260425_015740/checkpoints/epoch=006-train_vla_loss=2.820.ckpt`

## Useful Comparisons

These are valid configs for controlled comparisons, but they are not the next
main run.

- `dreamerv3_pixel_libero_10.yaml`
  - DreamerV3-style pixel world model.

- `dreamer_vla_libero_10_dreamerv3_pixel_actor.yaml`
  - DreamerVLA actor/critic with pretrained DreamerV3 pixel WM.

- `pretokenize_wm_libero_10_obs4096_minloss_rssm.yaml`
  - Original token input path with 4096 observation embedding and RSSM dynamics.

- `pretokenize_wm_libero_10_original_transdreamer.yaml`
  - Strict TransDreamer architecture ablation inside the DreamerVLA interface.

- `chameleon_latent_action_wm_libero_10.yaml`
  - Chameleon/LaDiWM-style world-model experiment.

## Historical / Do Not Use For Current Run

These configs are kept for reproducibility and ablation history. They should not
be used as the current Dreamer-token training config.

- `dreamer_vla_libero_10_dreamerv3_token_actor.yaml`
  - Older 20-epoch token actor config.
  - It has `run_wm_phase: true`, so it does not match the current decision to
    load the pretrained token WM and train actor/critic only.

- `dreamer_vla_libero_10.yaml`
  - Earlier base DreamerVLA config.

- `dreamer_vla_libero_10_transdreamer.yaml`
  - Earlier DreamerVLA + TransDreamer path.

- `dreamer_vla_libero_10_transdreamer_vlaactor.yaml`
  - Earlier VLA actor variant.

- `dreamer_vla_libero_10_rynn_native_wm_actor.yaml`
- `dreamer_vla_libero_10_rynn_hidden_recon_actor.yaml`
  - RynnVLA-related ablation configs.

- `pretokenize_sft_libero_10.yaml`
  - Debug/pretokenize SFT config, not part of the current Dreamer-token run.

- `pretokenize_wm_libero_10.yaml`
- `pretokenize_wm_libero_10_transdreamer.yaml`
- `pretokenize_wm_libero_10_v2.yaml`
- `pretokenize_wm_libero_10_warmup.yaml`
- `pretokenize_wm_libero_10_warmup_flat.yaml`
- `pretokenize_wm_libero_10_warmup_rollout.yaml`
- `pretokenize_wm_libero_10_dreamer_cnn.yaml`
- `pretokenize_wm_libero_10_dreamer_cnn_gauss.yaml`
- `pretokenize_wm_libero_10_dreamer_cnn_gauss_action.yaml`
- `pretokenize_wm_libero_10_dreamer_cnn_pure_vae.yaml`
- `pretokenize_wm_libero_10_dreamer_cnn_gauss_pure_vae.yaml`
- `pretokenize_wm_libero_10_obs4096_minloss.yaml`
- `pretokenize_wm_libero_10_obs4096_minloss_transdreamer_sumce.yaml`
  - WM architecture/loss ablations from the previous investigation. Keep for
    reproducing those results, but do not use as the current Dreamer-token
    policy training config.

## Output Layout

All active configs should write into one of the four output roots:

- `data/outputs/worldmodel`
- `data/outputs/vla`
- `data/outputs/dreamervla`
- `data/outputs/eval`

Old top-level output roots such as `pretokenize_wm`, `pretokenize_vla`,
`dreamer_vla`, `dreamer_vla_ablation`, `pure_vae`, and `eval_wm` should not be
reintroduced directly under `data/outputs`.
