# Route Reference

This file maps public experiments to their runner family and current status. Hydra remains the
source of truth; this table is a reader index.

## Mainline OpenVLA-OFT Cold Start

| Stage | Entry | Config / runner | Status |
| --- | --- | --- | --- |
| collect, Ray | `experiment=collect_rollouts_ray` | `ColdStartRayCollectRunner` | current |
| collect, no-Ray | `experiment=collect_rollouts_onetraj` | `CollectRolloutsRunner` | current |
| sync warmup + cotrain baseline | `experiment=online_cotrain_pipeline_oft_backbone_latent` | `OnlineCotrainPipelineRunner` | current baseline |
| async manual cotrain | `experiment=manual_cotrain_ray_oft_backbone_latent` | `ManualCotrainRayRunner` | current manual route |
| tiny manual smoke | `experiment=manual_cotrain_ray_tiny` | `ManualCotrainRayRunner` | local smoke |

Pipeline launcher:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=6 profile=multi_gpu cotrain_engine=async render_backend=egl
```

## Core Training Routes

| Family | Experiments | Runner |
| --- | --- | --- |
| VLA SFT | `vla_rynnvla_action_head`, `vla_sft_one_trajectory` | `VLASFTRunner` |
| OpenVLA-OFT SFT | `openvla_oft_hdf5`, `openvla_oft_hdf5_one_trajectory`, `openvla_oft_hdf5_one_trajectory_l1` | `OpenVLAOFTRunner` |
| world model | `world_model_wm_step`, `world_model_wm_chunk`, `oft_world_model_wm_chunk`, `oft_discrete_token_world_model_wm_chunk` | `LatentWMRunner` |
| classifier | `latent_classifier_libero_goal_chunk`, `oft_latent_classifier_chunk` | `LatentClassifierRunner` |
| DreamerVLA offline/online LUMOS | `dreamervla_rynn_wm_lumos`, `dreamervla_oft_wm_lumos`, `dreamervla_oft_discrete_token_wm_lumos` | `JointDreamerVLARunner` |
| eval | `eval_libero_vla` | `EmbodiedEvalRunner` |

## Optional Or Legacy Routes

| Experiment | Runner | Status |
| --- | --- | --- |
| `online_cotrain_ray_oft_backbone_latent` | `OnlineCotrainRayRunner` | optional legacy Ray route |
| `online_cotrain_ray_oft_action_hidden` | `OnlineCotrainRayRunner` | optional legacy Ray route |
| `online_cotrain_ray_world_model_env_tiny` | `OnlineCotrainRayRunner` | world-model-env smoke |
| `online_cotrain_ray_synthetic` | `OnlineCotrainRayRunner` | synthetic backend smoke |

Do not make legacy Ray routes the default path unless the active task explicitly asks for them.

## Config Ownership

- `configs/train.yaml` composes the top-level Hydra groups.
- `configs/experiment/*.yaml` selects a complete route.
- `configs/dreamervla/*.yaml` owns DreamerVLA/manual cotrain component wiring.
- `configs/scripts/coldstart_warmup_cotrain.yaml` owns launcher-level pipeline controls.
- `configs/task/*.yaml` owns LIBERO suite, checkpoint, image/history, and sidecar metadata.
