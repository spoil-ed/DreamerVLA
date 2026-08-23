from __future__ import annotations


def test_mainline_actor_classes_are_importable_from_split_modules() -> None:
    from dreamervla.algorithms.actor import (
        LatentToOpenVLAHiddenStateActor,
        VLAPolicy,
    )
    from dreamervla.algorithms.actor.latent_to_openvla_hidden_state_actor import (
        LatentToOpenVLAHiddenStateActor as SplitActor,
    )
    from dreamervla.algorithms.actor.vla_policy import VLAPolicy as SplitPolicy

    assert LatentToOpenVLAHiddenStateActor is SplitActor
    assert VLAPolicy is SplitPolicy


def test_removed_observation_actors_are_not_exported() -> None:
    import dreamervla.algorithms.actor as actor

    for name in (
        "LatentToHiddenTokenActor",
        "LatentToOpenVLADiscreteTokenActor",
        "OpenVLADiscreteTokenActor",
        "RynnVLAHiddenTokenActor",
        "VLAActionHeadActor",
    ):
        assert not hasattr(actor, name), name
