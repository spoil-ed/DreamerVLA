# OpenVLA-OFT on LIBERO-Goal

Model-on-dataset notes for the OpenVLA-OFT backbone on the LIBERO-Goal suite.
All shapes and attributes below were verified against artifacts on disk.

## Checkpoint formats

| Format | Example path | Action head | Detection |
| --- | --- | --- | --- |
| Component-wise L1 | `data/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650/` | `action_head--6650_checkpoint.pt` + `proprio_projector--*.pt` + LoRA-merged backbone | `action_head--*_checkpoint.pt` present → `l1` |
| Merged discrete LM-head | `data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1/` | LM head (vocab-tail action tokens), merged into backbone | no component files → `discrete` |

One-trajectory SFT routes: `openvla_oft_hdf5_one_trajectory` (discrete) and
`openvla_oft_hdf5_one_trajectory_l1` (L1, component checkpoints).

## Action hidden (Scheme A)

Both head formats expose the same layer: the last LM hidden states at the
56 action slots (8-step chunk × 7 dims), taken **before** the action head
(`_regression_or_discrete_prediction`). Per frame:

- `action_hidden_states`: `[56, 4096]`
- `obs_embedding` (WM input): flat `[229376]` (= 56 × 4096)

Sidecar attrs (`data/<task>.hdf5` file level):

| Attr | L1 | Discrete |
| --- | --- | --- |
| `action_head_type` | `oft_l1_regression` | `oft_discrete_token` |
| `obs_hidden_source` | `action_query` | `action_query` |
| `history` / `include_state` | 2 / true (6650 recipe) | 1 / false (single view, no proprio) |
| `prompt_style` / `rotate_images_180` | `vla_policy` / true | `vla_policy` / true |

Reference sidecar:
`data/processed_data/libero_goal_no_noops_t_256_oft_hdf5_6650_legacy_action_hidden_vla_policy_h2/`.

This mirrors the RynnVLA-002 downstream contract: the WM observation tokens are
action slots.  OFT differs in token count and width (`56 × 4096` instead of
RynnVLA's `35 × 1024`), but the dataset/WM/classifier/DreamerVLA interface is
the same.

## Input tokens (Scheme B)

Scheme B writes current-frame projected vision patch tokens from the OFT
vision tower before the action-slot prediction branch.  For the default L1
two-view recipe:

- token source: `vla._process_vision_features(...)`
- views: current `agentview_rgb` + current `eye_in_hand_rgb`
- `obs_embedding` (WM input): flat `[2097152]` (= 512 × 4096)
- sidecar attrs: `obs_hidden_source=input_token_embedding`,
  `action_head_type=oft_l1_regression`, `history=2`

For downloaded discrete one-trajectory checkpoints with a single view, set
`OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb`; the token count becomes `256`.
B tokens are frame-level visual observations, not action slots.  The WM handles
actions through its action input, and DreamerVLA uses `LatentToActionHiddenActor`
when a continuous L1 output head is available or explicitly configured.

## Extraction

```bash
# L1 (auto-detected):
TASK=libero_goal OFT_CKPT=data/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650 \
bash scripts/preprocess/35_oft_action_hidden.sh

# Discrete one-trajectory download:
TASK=libero_goal \
OFT_CKPT=data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
OFT_POLICY_MODE=discrete OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb \
bash scripts/preprocess/35_oft_action_hidden.sh

# Scheme B input tokens:
TASK=libero_goal OFT_LATENT_SCHEME=input_tokens \
OFT_CKPT=data/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650 \
bash scripts/preprocess/35_oft_action_hidden.sh
```

First run on a new checkpoint: smoke with `--max-files 1 --max-demos-per-file 2`
by invoking the module directly before committing GPU hours.

## Downstream chain (unchanged by head format)

WM consumes `token_count=56 × token_dim=4096` per frame
(`task.openvla_oft.*` in `configs/task/libero_goal.yaml`):

```bash
bash scripts/train_wm.sh --config oft_world_model_dinowm_chunk --task libero_goal
bash scripts/train_wm.sh --config oft_world_model_dinowm_chunk_input_tokens --task libero_goal
# discrete sidecars: override ckpt_path / action_hidden_dir /
# expected_action_head_type=oft_discrete_token / expected_history=1 /
# expected_include_state=false (see SETUP.md)
```

Classifier: `oft_latent_classifier_chunk` · DreamerVLA:
`dreamervla_oft_dino_wm_wmpo_outcome` · Scheme B:
`oft_latent_classifier_chunk_input_tokens` and
`dreamervla_oft_dino_wm_wmpo_outcome_input_tokens` · Eval:
`scripts/eval/launch_openvla_oft_official_libero_eval.sh` (policy mode
auto-detected the same way).
