# Actor Module Layout

Actor implementations are split by class while `src.models.vla_actor` remains a
compatibility import path for existing configs.

| File | Main contents |
| --- | --- |
| `base_actor.py` | `BaseActor` and shared Gaussian action distribution helper. |
| `vla_action_head_actor.py` | `VLAActionHeadActor`, the full VLA ActionHead reuse path. |
| `pi0_action_hidden_actor.py` | `Pi0ActionHiddenActor`, the action-hidden DreamerVLA actor path. |
| `__init__.py` | Public actor exports. |

Preferred new imports:

```python
from src.models.actor import Pi0ActionHiddenActor, VLAActionHeadActor
```

Existing Hydra targets such as `src.models.vla_actor.Pi0ActionHiddenActor` are
kept valid by the compatibility module.
