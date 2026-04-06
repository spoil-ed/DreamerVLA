# Dreamer-VLA

DreamerVLA prototype for combining:

- `RynnVLA-001` as the VLA encoder / action prior
- a bottleneck module for extracting compact physical state
- `dreamerv3-main` as the world model backbone
- a controller or planner that selects actions from imagined rollouts

## Current Status

The repository now contains a working prototype that closes the main
training loop:

- multimodal encoder -> hidden state
- Gaussian policy with `ref / old / new` views
- simple latent dynamics world model
- PPO-style actor update with grouped normalized advantages
- a runnable training entrypoint

The codebase currently references two external roots in
[`configs/base.yaml`](configs/base.yaml):

- `RynnVLA-001`
- `dreamerv3-main`

## Layout

```text
Dreamer-VLA/
├── configs/
│   ├── base.yaml
│   └── ppo_trainer.yaml
├── docs/
│   └── architecture.md
├── scripts/
│   └── train.py
├── src/
│   ├── algorithms/
│   │   └── ppo_grpo.py
│   ├── models/
│   │   ├── critic.py
│   │   ├── vla_policy.py
│   │   ├── world_model/
│   │   └── vla_encoder/
│   └── workspace/
│       ├── base_workspace.py
│       └── dreamer_vla_workspace.py
└── pretrained_models/
```

## Directory Roles

- `configs/`: experiment, model, trainer, and external dependency paths
- `docs/`: high-level design notes for the Dreamer-VLA pipeline
- `scripts/`: runnable demo entrypoints
- `src/algorithms/`: PPO / GRPO-style loss utilities
- `src/models/`: policy, world model, critic, and the existing encoder code
- `src/workspace/`: workspace entry logic, training state, and top-level training loop
- `pretrained_models/`: local placeholder for downloaded checkpoints

## Planned Pipeline

1. Encode `(image, proprio, text)` into a hidden state.
2. Project the hidden state into a latent `z_t`.
3. Predict `z_{t+1}` with a simple dynamics model.
4. Score grouped candidate actions with the reward head.
5. Update the actor with PPO loss plus ref-policy KL regularization.

## Notes

- The encoder implementation under `src/models/vla_encoder/` is kept intact.
- The current demo target is `python scripts/train.py`.
