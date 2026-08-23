# Route Reference

This file maps release experiments to their runner family. Hydra remains the
source of truth; this table is a reader index.

## Mainline OpenVLA-OFT Cold Start

| Stage | Entry | Config / runner | Status |
| --- | --- | --- | --- |
| collect | `experiment=collect_rollouts` | `RolloutCollectionRunner` | current; Ray backend |
| world-model warmup | `experiment=dreamer-wm` or `dino-wm` | WM runner selected by recipe | current |
| classifier warmup | `experiment=classifier_official_upper_bound` | `SuccessClassifierTrainingRunner` | current |
| cotrain | `experiment=openvla_libero` | `DreamerRunner` | current; frozen WM/CLS; Ray backend |
| eval | `experiment=eval_cotrain` | `LIBEROVLAEvaluationRunner` | current |

Collection:

```bash
python -m dreamervla.train \
  experiment=collect_rollouts task=openvla_onetraj_coldstart_libero
```

The reduced shell surface uses `scripts/experiments/cotrain/train.sh` and
`scripts/experiments/cotrain/eval.sh` for the trainable WM/CLS route.

## Supporting Training Entrypoints

| Family | Experiments | Runner |
| --- | --- | --- |
| full staged cotrain | `openvla_onetraj_libero_cotrain` | `CotrainRunner` |
| world model | `wm_full_dataset_train` | `WorldModelTrainingRunner` |
| eval | `eval_libero_vla` | `LIBEROVLAEvaluationRunner` |

## Config Ownership

- `configs/train.yaml` composes the top-level Hydra groups.
- `configs/experiment/*.yaml` selects a complete route.
- `configs/dreamervla/*.yaml` owns DreamerVLA/manual cotrain component wiring.
- `configs/classifier/dreamer-cls.yaml` owns the classifier role recipe.
- `configs/task/*.yaml` owns LIBERO metadata and the concrete classifier model/dataset/input contract.
