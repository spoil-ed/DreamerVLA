from __future__ import annotations


def test_actor_classes_are_importable_from_split_modules() -> None:
    from dreamer_vla.models.actor import Pi0ActionHiddenActor, VLAActionHeadActor
    from dreamer_vla.models.actor import VLAPolicy
    from dreamer_vla.models.actor.pi0_action_hidden_actor import (
        Pi0ActionHiddenActor as SplitPi0ActionHiddenActor,
    )
    from dreamer_vla.models.actor.vla_policy import VLAPolicy as SplitVLAPolicy
    from dreamer_vla.models.actor.vla_action_head_actor import (
        VLAActionHeadActor as SplitVLAActionHeadActor,
    )
    from dreamer_vla.models.vla_actor import Pi0ActionHiddenActor as LegacyPi0ActionHiddenActor
    from dreamer_vla.models.vla_actor import VLAActionHeadActor as LegacyVLAActionHeadActor
    from dreamer_vla.models.vla_policy import VLAPolicy as LegacyVLAPolicy

    assert Pi0ActionHiddenActor is SplitPi0ActionHiddenActor
    assert VLAActionHeadActor is SplitVLAActionHeadActor
    assert VLAPolicy is SplitVLAPolicy
    assert LegacyPi0ActionHiddenActor is SplitPi0ActionHiddenActor
    assert LegacyVLAActionHeadActor is SplitVLAActionHeadActor
    assert LegacyVLAPolicy is SplitVLAPolicy
