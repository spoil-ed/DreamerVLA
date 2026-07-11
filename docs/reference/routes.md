# Route Reference

This file maps release experiments to their runner family. Hydra remains the
source of truth; this table is a reader index.

## Mainline OpenVLA-OFT Cold Start

| Stage | Entry | Config / runner | Status |
| --- | --- | --- | --- |
| collect, Ray | `experiment=collect_rollouts_ray` | `ColdStartRayCollectRunner` | current |
| collect, no-Ray | `experiment=collect_rollouts_onetraj` | `CollectRolloutsRunner` | current |
| sync warmup + cotrain baseline | `experiment=openvla_onetraj_libero_cotrain_noray` | `OnlineCotrainPipelineRunner` | current baseline |
| async manual cotrain | `experiment=openvla_onetraj_libero_cotrain_ray` | `ManualCotrainRayRunner` | current manual route |

Pipeline launcher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal profile=multi_gpu cotrain_engine=async render_backend=osmesa
```

## Supporting Training Entrypoints

| Family | Experiments | Runner |
| --- | --- | --- |
| world model | `wm_full_dataset_train` | `OnlineCotrainPipelineRunner` warmup path |
| eval | `eval_libero_vla` | `EmbodiedEvalRunner` |

## Config Ownership

- `configs/train.yaml` composes the top-level Hydra groups.
- `configs/experiment/*.yaml` selects a complete route.
- `configs/dreamervla/*.yaml` owns DreamerVLA/manual cotrain component wiring.
- `configs/scripts/coldstart_warmup_cotrain.yaml` owns launcher-level pipeline controls.
- `configs/task/*.yaml` owns LIBERO suite, checkpoint, image/history, and sidecar metadata.
