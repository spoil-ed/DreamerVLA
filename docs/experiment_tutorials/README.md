# Experiment Tutorials

These tutorials are short end-to-end recipes. They start from a clean checkout,
then go through install, download, preprocessing, training, and final LIBERO
eval.

Use Hydra overrides for normal changes:

```bash
gpus=0,1 ngpu=2 batch_size=16 num_workers=4 max_steps=1000 out_dir=/tmp/run
```

The shell launchers are intentionally thin. They set project/data roots and
call the Hydra launcher; all experiment choice is in `experiment=...` and
normal Hydra keys.

## Recipes

| Recipe | Latent | Main configs |
| --- | --- | --- |
| [RynnVLA Scheme A](rynnvla_action_hidden_libero_goal.md) | action-query hidden tokens | `world_model_dinowm_chunk`, `dreamervla_rynn_dino_wm_wmpo_outcome` |
| [RynnVLA Scheme B](rynnvla_input_token_libero_goal.md) | current-frame Chameleon input tokens | `world_model_dinowm_chunk_input_tokens`, `dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens` |
| [OpenVLA-OFT Scheme A](openvla_oft_action_hidden_libero_goal.md) | OFT action-slot hidden tokens | `oft_world_model_dinowm_chunk`, `dreamervla_oft_dino_wm_wmpo_outcome` |
| [OpenVLA-OFT Scheme B](openvla_oft_input_token_libero_goal.md) | current-frame projected vision tokens | `oft_world_model_dinowm_chunk_input_tokens`, `dreamervla_oft_dino_wm_wmpo_outcome_input_tokens` |

Scheme A is the RynnVLA-002 contract: the WM token axis is an action-slot axis.
OFT Scheme A uses the same downstream contract, with different token shape.
Scheme B is frame-level visual input tokens; the WM receives actions through
its action input and DreamerVLA uses a bridge actor to produce action slots.

Classifier and WMPO training need both success and failure rollout corpora. The
standard LIBERO download gives success demos. If you do not have failure demos
and matching sidecars yet, stop after WM training or use the actor-critic route
where applicable.
