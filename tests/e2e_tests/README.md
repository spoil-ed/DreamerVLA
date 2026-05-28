# End-to-end tests

End-to-end tests drive a full training or evaluation route. Unlike `unit_tests/`,
they may spawn subprocesses, instantiate the real LIBERO env, load real
checkpoints under `data/ckpts/`, or call `python -m src.cli.train` with a
mainline route YAML — and they are expected to take minutes rather than
seconds.

Layout per the project guide ([../../CLAUDE.md](../../CLAUDE.md)):

```text
e2e_tests/
  vla/         # VLA SFT end-to-end
  wm/          # World model end-to-end
  dreamervla/  # Joint DreamerVLA end-to-end
  oft/         # OpenVLA-OFT end-to-end
  classifier/  # LatentSuccessClassifier end-to-end
  <route>/*.yaml  # e2e Hydra configs, one folder per route
```

This directory is currently empty — all 28 existing test files were classified
as `unit_tests/` during the 2026-05-27 sweep (no full-route execution detected,
all use synthetic tensors / mocks / tmp_path). Add e2e tests here when a
regression genuinely requires real env or real ckpts to catch.
