# Actor Module Layout

Actor implementations are split by class while top-level modules such as
`dreamer_vla.models.vla_actor` and `dreamer_vla.models.vla_policy` remain
compatibility import paths for existing code.

| File | Main contents |
| --- | --- |
| `base_actor.py` | `BaseActor` and shared Gaussian action distribution helper. |
| `vla_action_head_actor.py` | `VLAActionHeadActor`, the full VLA ActionHead reuse path. |
| `rynnvla_action_hidden_actor.py` | `RynnVLAActionHiddenActor`, the action-hidden DreamerVLA actor path. |
| `vla_policy.py` | `SharedObservationEmbedding` and `VLAPolicy`. |
| `__init__.py` | Public actor exports. |

Preferred new imports:

```python
from dreamer_vla.models.actor import RynnVLAActionHiddenActor, VLAActionHeadActor, VLAPolicy
```

Compatibility exports remain available through `dreamer_vla.models.vla_actor`.
