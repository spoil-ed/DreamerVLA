# RynnVLA + LIBERO-Goal

RynnVLA is a secondary historical route. The active release script surface no
longer ships standalone VLA/WM training launchers, and the current mainline is
the OpenVLA-OFT cold-start workflow in
[OpenVLA_Onetraj_LIBERO.md](OpenVLA_Onetraj_LIBERO.md).

For active work, use the OpenVLA-OFT recipe and the current script registry:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal dry_run=true
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal dry_run=true
bash scripts/eval_libero_vla.sh --help
```

The role-based RynnVLA WM route name remains:

```bash
python -m dreamervla.train experiment=world_model_chunk task=rynnvla_libero
python -m dreamervla.train experiment=dreamervla_rynn_wm_lumos task=rynnvla_libero
python -m dreamervla.train experiment=dreamervla_rynn_wm_actor_critic task=rynnvla_libero
```

RynnVLA artifacts and older standalone training wrappers may still exist for
reproducibility, but they are not part of the current publishable tutorial path.
