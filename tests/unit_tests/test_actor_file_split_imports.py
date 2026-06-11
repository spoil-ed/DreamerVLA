from __future__ import annotations


def test_actor_classes_are_importable_from_split_modules() -> None:
    from dreamer_vla.models.actor import RynnVLAActionHiddenActor, VLAActionHeadActor, VLAPolicy
    from dreamer_vla.models.actor.rynnvla_action_hidden_actor import (
        RynnVLAActionHiddenActor as SplitRynnVLAActionHiddenActor,
    )
    from dreamer_vla.models.actor.vla_action_head_actor import (
        VLAActionHeadActor as SplitVLAActionHeadActor,
    )
    from dreamer_vla.models.actor.vla_policy import VLAPolicy as SplitVLAPolicy

    assert RynnVLAActionHiddenActor is SplitRynnVLAActionHiddenActor
    assert VLAActionHeadActor is SplitVLAActionHeadActor
    assert VLAPolicy is SplitVLAPolicy
