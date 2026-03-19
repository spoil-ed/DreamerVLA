# Dreamer-VLA

Research scaffold for combining:

- `RynnVLA-001` as the state encoder / action prior
- a bottleneck module for extracting compact physical state
- `dreamerv3-main` as the world model backbone
- a planner that selects the best action from imagined rollouts

## Layout

```text
Dreamer-VLA/
├── configs/
├── docs/
├── scripts/
├── src/dreamer_vla/
└── tests/
```

This repository is currently initialized as a minimal project skeleton.
