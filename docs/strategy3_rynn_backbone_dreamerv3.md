# Strategy 3: Pixel Observation + RynnVLA Backbone + DreamerV3 RSSM

## Core Answer

Yes.  The observation should remain pixel, and the DreamerV3 encoder slot should
be the frozen RynnVLA-002 backbone.  The downstream world model remains the same
DreamerV3 structure used by the previous pixel/token runs:

```text
pixel obs_t
  -> frozen RynnVLA-002 image/text backbone encoder
  -> 4096-d hidden observation embedding
  -> DreamerV3 RSSM observe/imagine
  -> pixel decoder + reward head + continue head
```

This is different from pre-extracting latents and training a separate flow
model.  Here, the WM still receives pixels as observations.

## Loss Alignment

The loss is intentionally aligned with `DreamerV3PixelWorldModel`.  The only
changed line is the observation encoder source:

```text
pixel_dreamerv3:
  images -> DreamerV3PixelEncoder -> RSSM -> pixel decoder

rynn_backbone_dreamerv3:
  images -> frozen RynnVLA backbone -> RSSM -> pixel decoder
```

The reconstruction target is still RGB pixels, not the 4096-d Rynn hidden
context.  The active metrics are the pixel metrics:

```text
loss = rec_scale * rec_loss
     + dyn_scale * dyn_loss
     + rep_scale * rep_loss
     + rew_scale * reward_loss
     + con_scale * continue_loss

rec_loss = mean(sum((recon_rgb - target_rgb)^2 over C,H,W))
image_mse = mean((recon_rgb - target_rgb)^2)
image_psnr = -10 * log10(image_mse)
```

`dyn_loss`, `rep_loss`, `reward_loss`, `continue_loss`, `dyn_entropy`, and
`rep_entropy` are computed by the same RSSM/reward/continue code path as the
pixel baseline.

## WM-Only Stage

Implemented entry points:

- Config: `configs/rynn_backbone_dreamerv3_pixel_wm_libero_goal.yaml`
- Workspace: `src/workspace/rynn_backbone_dreamerv3_wm_workspace.py`
- WM class: `src.models.world_model.dreamerv3_torch.DreamerV3PixelRynnBackboneWorldModel`
- Script: `scripts/train_rynn_backbone_dreamerv3_wm.sh`
- Data horizon: `sequence_length=32`, `stride=1`, matching the pixel
  DreamerV3 setup.  The RynnVLA encoder is chunked with
  `training.encoder_chunk_size=8` to keep the frozen backbone pass practical.

Smoke:

```bash
WM_SMOKE=1 WM_SMOKE_STEPS=1 CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
  bash scripts/train_rynn_backbone_dreamerv3_wm.sh
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 BATCH_SIZE=1 GRAD_ACCUM=2 \
  bash scripts/train_rynn_backbone_dreamerv3_wm.sh
```

## Policy Stage

After WM loss is sane, reuse the RynnVLA action head through the existing
`VLAActionHeadActor` in identity mode:

```text
RSSM imagined state
  -> actor adapter or action-context head
  -> frozen or lightly tuned RynnVLA action head
  -> action chunk
```

For the first test, freeze the Rynn backbone and action head, and only train the
RSSM, pixel decoder, reward/continue heads, critic, and possibly a tiny actor
projection if the hidden scale needs calibration.
