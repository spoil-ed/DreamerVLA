# WM Training Routes

Active world-model and DreamerVLA routes in this repo. Older pooled-hidden,
pretokenized token WM, semantic bottleneck, and scalar-reward actor branches
have been removed from the public training surface.

## Current Mainline

### 1. pi0 Action-Hidden DreamerV3 WM

Config:

- `configs/rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed.yaml`

Route:

```text
LIBERO obs + language
-> pi0-query VLA backbone/action-query block
-> action-query hidden after the action-head transformer
-> flatten [H, 1024] to obs_embedding [H*1024]
-> DreamerV3 RSSM posterior / transition
-> hidden reconstruction + twohot reward + continue + optional image reconstruction
```

Use:

```bash
PIPELINE_STAGE=preprocess bash scripts/run_pi0_query_hidden_pipeline.sh
PIPELINE_STAGE=wm bash scripts/run_pi0_query_hidden_pipeline.sh
```

### 2. pi0 Action-Hidden DreamerVLA Actor

Config:

- `configs/dreamer_vla_libero_goal_pi0_action_hidden_head_actor.yaml`

Route:

```text
DreamerV3 RSSM feature
-> hidden decoder reconstructs pi0 action hidden
-> Pi0ActionHiddenActor
-> VLA output projection
-> action
```

Use:

```bash
bash scripts/run_pi0_action_hidden_reconstruct_actor.sh
```

## Retained Baselines

### Pixel DreamerV3 WM

- Config: `configs/dreamerv3_pixel_libero_goal.yaml`
- Runner: `dreamer_vla.runners.PixelWMRunner`
- Script: `scripts/train_dreamerv3_pixel.sh`

### Token DreamerV3 WM

- Config: `configs/dreamerv3_token_libero_goal.yaml`
- Runner: `dreamer_vla.runners.TokenWMRunner`
- Script: `scripts/train_dreamerv3_token.sh`

### Chameleon / LaDiWM-Style WM

- Config: `configs/chameleon_latent_action_wm_libero_goal.yaml`
- Runner: `dreamer_vla.runners.ChameleonLatentWMRunner`
- Script: `scripts/train_chameleon_ladiwm_wm.sh`

## Removed Routes

The old pooled-hidden DreamerV3 WM, pretokenized TransDreamer/TSSM token WM,
semantic bottleneck WM, old scalar-reward actor configs, and their diagnostic
helpers were deleted. Existing checkpoints from those branches are historical
artifacts; new runs should use one of the routes above.
