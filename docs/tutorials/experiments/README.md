# Experiment tutorials

Step-only end-to-end recipes (clean checkout → install → download → preprocess →
train → eval). **All background, rationale, memory/OOM, WM sizing and logging notes
are in [EXPLAINED.md](EXPLAINED.md).** Parameter reference:
[PARAMETERS.md](../../PARAMETERS.md).

Normal changes are Hydra overrides, e.g.
`gpus=0,1 ngpu=2 batch_size=16 num_workers=4 num_epochs=20 out_dir=/tmp/run`.

## Recipes

| Pipeline | Hydra `task=` | Main configs |
| --- | --- | --- |
| [RynnVLA_LIBERO](RynnVLA_LIBERO.md) | `rynnvla_libero` | `world_model_dinowm_chunk`, `dreamervla_rynn_dino_wm_lumos` |
| [OpenVLA one-traj](OpenVLA_Onetraj_LIBERO.md) | `openvla_onetraj_libero` | `oft_discrete_token_world_model_dinowm_chunk`, `dreamervla_oft_discrete_token_dino_wm_lumos` |
| [OFT action-hidden WM (Scheme A)](OpenVLA_Onetraj_LIBERO_action_hidden_world_model.md) | `openvla_onetraj_libero` | `oft_world_model_dinowm_chunk`, `oft_latent_classifier_chunk`, `dreamervla_oft_dino_wm_lumos`, `online_cotrain_oft_action_hidden` |
| [OFT backbone-latent WM (Scheme 1)](OpenVLA_Onetraj_LIBERO_backbone_latent_world_model.md) | `openvla_onetraj_libero` | `oft_world_model_dinowm_chunk_input_tokens`, `dreamervla_oft_dino_wm_lumos_input_tokens`, `online_cotrain_oft_backbone_latent` |
| [Cold-start rollout collection](OpenVLA_Onetraj_LIBERO_coldstart_rollout_collection.md) | `openvla_onetraj_coldstart_libero` | `collect_rollouts_onetraj`, `oft_discrete_token_world_model_dinowm_chunk` |
| [Cold-start collect + warmup + cotrain](OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md) | `openvla_onetraj_coldstart_libero` | `collect_rollouts_ray`, `online_cotrain_pipeline_oft_action_hidden` |
| [Ray/manual cotrain backend](../../../spec/02_ray.md) | synthetic / gated real smoke | `manual_cotrain_ray_*`, legacy `online_cotrain_ray_*` |

The `task=` token is snake_case; on-disk data artifacts keep their historical
`task.artifact_name` directories (e.g. `OpenVLA_Onetraj_LIBERO_libero_goal`), so paths
inside the commands mix the two — this is intentional (see EXPLAINED.md).

## Validation notes

- [RLinf-aligned LIBERO rollout](../../archive/plans/RLinf_aligned_LIBERO_rollout_execution_plan.md) — the
  OpenVLA-OFT / RLinf action contract and the shared rollout core.
- [Architecture index](../../../spec/00_overview.md) — target-mainline group topology.
- [Complete loop](../../../spec/01_complete_loop.md) — collect, warmup, cotrain,
  trajectory PPO, checkpoint, and eval coverage.
- [Ray implementation](../../../spec/02_ray.md) — worker groups, placement, channels, and current implementation status.
- [Cold-start cotrain bracket checklist](../../reports/audits/online_cotrain_bracket_resolution_checklist.md) —
  focused review checklist for bracket/order issues in the cotrain path.
