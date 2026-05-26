# Config Registry

Hydra configs are now grouped by training route. Shell launchers should choose
one concrete config and keep operational overrides small: GPUs, batch size,
checkpoint paths, output tags, and smoke-test limits.

## Formal Entrypoints

| Stage | Script | Default Config |
| --- | --- | --- |
| VLA training | `scripts/train_vla.sh` | `vla_pi0_query` |
| WM training | `scripts/train_wm.sh` | `world_model_rssm_step` |
| DreamerVLA training | `scripts/train_dreamervla.sh` | `dreamervla_pi0_action_hidden_head_actor` |

## Formal Configs

| Route | Config |
| --- | --- |
| VLA | `vla_pi0_query` |
| WM RSSM step | `world_model_rssm_step` |
| WM DINO step | `world_model_dinowm_step` |
| WM DINO chunk | `world_model_dinowm_chunk` |
| DreamerVLA RSSM | `dreamervla_pi0_action_hidden_head_actor` |
| DreamerVLA PPO, DINO-WM step | `dreamervla_rynn_dino_wm_actor_critic` |
| DreamerVLA PPO, DINO-WM chunk/outcome | `dreamervla_rynn_dino_wm_wmpo_outcome` |

Formal configs use Hydra defaults to include the task config:

```yaml
defaults:
  - _self_
  - /task: libero_goal
```

Keep concrete dataset task paths, horizons, sidecar expectations, and
task-specific dimensions in `task/*.yaml`. The training configs define the
workspace, model, optimizer, and algorithm route.

## Task Configs

The task folder contains concrete dataset task definitions only:

```text
task/libero_goal.yaml
task/libero_object.yaml
task/libero_spatial.yaml
task/libero_10.yaml
```

Switch tasks with Hydra, for example:

```bash
bash scripts/train_vla.sh task=libero_object
WM_KIND=dinowm bash scripts/train_wm.sh task=libero_spatial
```

## Archive

Unconfirmed or historical configs live under:

```text
archive/uncertain_configs/
archive/libero10_legacy/
```

Do not use archived configs as defaults without checking paths, checkpoint
compatibility, and sidecar schema.
