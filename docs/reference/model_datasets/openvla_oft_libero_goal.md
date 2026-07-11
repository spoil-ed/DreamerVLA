# OpenVLA-OFT on LIBERO-Goal

Model-on-dataset notes for the OpenVLA-OFT backbone on the LIBERO-Goal suite.

## Checkpoint Formats

| Format | Example path | Detection |
| --- | --- | --- |
| Merged discrete LM-head | `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1/` | merged checkpoint directory |

Component-wise L1/action-query checkpoints are rejected by every active route.

## Hidden-Token Sidecars

The current one-trajectory route uses projected current-frame vision tokens for
WM and classifier warmup. The sidecar contract is stored in
`task.openvla_oft.hidden_token.*`:

- `token_count`: 256 for the single mainline input image.
- `token_dim`: 4096.
- `wm_obs_dim`: 1048576 (`256 * 4096`).
- `expected_obs_hidden_source`: `hidden_token`.
- `expected_prompt_style`: `vla_policy`.

Reference directory:

```text
data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_hidden_token_vla_policy_h1/
```

## Extraction

```bash
TASK=libero_goal \
OFT_CKPT=data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb \
bash scripts/preprocess/35_oft_hidden_token.sh
```

## Downstream Chain

Use the cold-start cotrain launcher or the full-replay WM warmup script:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=8 profile=multi_gpu
bash scripts/experiments/world_model_training/train.sh
```
