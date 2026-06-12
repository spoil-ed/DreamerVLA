# RynnVLA on LIBERO-Goal

Model-on-dataset notes for the RynnVLA backbone (Chameleon + legacy action
head) on the LIBERO-Goal suite. All shapes verified against artifacts on disk.

## Checkpoint and assets

| Asset | Path |
| --- | --- |
| VLA checkpoint | `data/checkpoints/VLA_model_256/libero_goal/` |
| Lumina backbone/tokenizer | `data/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768/` |
| Chameleon text tokenizer / VQGAN | `data/checkpoints/chameleon/tokenizer/` |

One-trajectory SFT route: `vla_sft_one_trajectory`
(`dataset.trajectory_offset` selects the demo).

## Action hidden (Scheme A)

The legacy `RynnVLAActionHead` appends 35 learnable action-query tokens
(5-step chunk × 7 dims) to the context; their backbone outputs are projected
to 1024 dims. Per frame:

- `obs_embedding` (WM input): flat `[35840]` (= 35 × 1024)
- `action_hidden_states`: per-step query hidden for the actor

Sidecar attrs expected by the WM route
(`task.legacy_action_hidden.*`): `action_head_type=legacy`,
`obs_hidden_source=action_query`, `prompt_style=vla_policy`, `history=2`,
`include_state=true`, `rotate_images_180=true`.

Reference sidecar:
`data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2/`.

## Extraction

```bash
TASK=libero_goal bash scripts/preprocess/30_action_hidden.sh
# or as part of: bash scripts/preprocess/prepare_libero_data.sh
```

## Downstream chain

WM consumes `token_count=35 × token_dim=1024` per frame
(`task.legacy_action_hidden.*` in `configs/task/libero_goal.yaml`):

```bash
CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal
```

Classifier: `latent_classifier_libero_goal_chunk` · DreamerVLA:
`dreamervla_rynn_dino_wm_wmpo_outcome` / `_actor_critic` · Eval:
`bash scripts/eval_libero_vla.sh`.

## Workflow verification (2026-06-12, CPU interface level)

Verified against on-disk artifacts: assets present (VLA ckpt, Lumina,
Chameleon tokenizer/VQGAN); one-trajectory SFT dataset instantiates from
its route config (133 samples, `wm_action [5,7]`); sidecar contract
`obs_embedding [T, 35840]`; chunk-WM constructed from config (58.3M params)
and ran `loss()` on a real batch (chunk + rollout + reward terms); classifier
dataset paired 433 success / 67 failure latent demos (`[8, 35840]` windows);
DreamerVLA and eval routes compose. GPU-bound steps (SFT training, sidecar
re-extraction, joint training, sim eval) not executed in that pass.
`*_marked_t_256` is a regenerable intermediate; `*_metainfo.json` is not
referenced by any active config or module.
