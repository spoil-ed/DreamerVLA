# End-to-end tests

End-to-end tests drive a full training or evaluation route. Unlike `unit_tests/`,
they may spawn subprocesses, instantiate the real LIBERO env, load real
checkpoints under `data/checkpoints/`, or call `python -m dreamervla.train` with a
mainline route YAML — and they are expected to take minutes rather than
seconds.

The current suite uses a flat, stage-prefixed layout:

```text
e2e_tests/
  test_s1*.py                 # Ray cluster, worker-group, and channel checks
  test_s2*.py                 # env/replay integration
  test_s3*.py                 # inference-worker integration
  test_s4*.py                 # learner and weight-sync integration
  test_s5*.py                 # learner parity
  test_s6*.py                 # cold-start and real OpenVLA-OFT collection
  test_cotrain_smoke.py       # cotrain subprocess smoke
  test_scheduler_ray_smoke.py # scheduler smoke
  test_world_model_env_ray_smoke.py
```

These tests are opt-in and may require Ray, CUDA, LIBERO, real checkpoints, or
explicit environment gates. Keep dependency-free synthetic coverage under
`tests/unit_tests/`; add coverage here when a regression requires a real process,
distributed backend, environment, or checkpoint boundary to catch.
