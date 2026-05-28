from __future__ import annotations


def test_actor_classes_are_importable_from_split_modules() -> None:
    from src.models.actor import Pi0ActionHiddenActor, VLAActionHeadActor
    from src.models.actor.pi0_action_hidden_actor import (
        Pi0ActionHiddenActor as SplitPi0ActionHiddenActor,
    )
    from src.models.actor.vla_action_head_actor import (
        VLAActionHeadActor as SplitVLAActionHeadActor,
    )
    from src.models.vla_actor import Pi0ActionHiddenActor as LegacyPi0ActionHiddenActor
    from src.models.vla_actor import VLAActionHeadActor as LegacyVLAActionHeadActor

    assert Pi0ActionHiddenActor is SplitPi0ActionHiddenActor
    assert VLAActionHeadActor is SplitVLAActionHeadActor
    assert LegacyPi0ActionHiddenActor is SplitPi0ActionHiddenActor
    assert LegacyVLAActionHeadActor is SplitVLAActionHeadActor
