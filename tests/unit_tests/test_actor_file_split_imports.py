from __future__ import annotations


def test_actor_classes_are_importable_from_split_modules() -> None:
    from dreamervla.models.actor import (
        LatentToActionHiddenActor,
        OpenVLADiscreteTokenActor,
        RynnVLAActionHiddenActor,
        VLAActionHeadActor,
        VLAPolicy,
    )
    from dreamervla.models.actor.openvla_discrete_token_actor import (
        OpenVLADiscreteTokenActor as SplitOpenVLADiscreteTokenActor,
    )
    from dreamervla.models.actor.latent_to_action_hidden_actor import (
        LatentToActionHiddenActor as SplitLatentToActionHiddenActor,
    )
    from dreamervla.models.actor.rynnvla_action_hidden_actor import (
        RynnVLAActionHiddenActor as SplitRynnVLAActionHiddenActor,
    )
    from dreamervla.models.actor.vla_action_head_actor import (
        VLAActionHeadActor as SplitVLAActionHeadActor,
    )
    from dreamervla.models.actor.vla_policy import VLAPolicy as SplitVLAPolicy

    assert LatentToActionHiddenActor is SplitLatentToActionHiddenActor
    assert OpenVLADiscreteTokenActor is SplitOpenVLADiscreteTokenActor
    assert RynnVLAActionHiddenActor is SplitRynnVLAActionHiddenActor
    assert VLAActionHeadActor is SplitVLAActionHeadActor
    assert VLAPolicy is SplitVLAPolicy
