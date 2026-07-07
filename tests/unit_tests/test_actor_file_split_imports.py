from __future__ import annotations

from pathlib import Path


def test_actor_classes_are_importable_from_split_modules() -> None:
    from dreamervla.algorithms.actor import (
        LatentToActionHiddenActor,
        LatentToOpenVLADiscreteTokenActor,
        LatentToOpenVLAHiddenStateActor,
        OpenVLADiscreteTokenActor,
        RynnVLAActionHiddenActor,
        VLAActionHeadActor,
        VLAPolicy,
    )
    from dreamervla.algorithms.actor.latent_to_action_hidden_actor import (
        LatentToActionHiddenActor as SplitLatentToActionHiddenActor,
    )
    from dreamervla.algorithms.actor.latent_to_openvla_discrete_token_actor import (
        LatentToOpenVLADiscreteTokenActor as SplitLatentToOpenVLADiscreteTokenActor,
    )
    from dreamervla.algorithms.actor.latent_to_openvla_hidden_state_actor import (
        LatentToOpenVLAHiddenStateActor as SplitLatentToOpenVLAHiddenStateActor,
    )
    from dreamervla.algorithms.actor.openvla_discrete_token_actor import (
        OpenVLADiscreteTokenActor as SplitOpenVLADiscreteTokenActor,
    )
    from dreamervla.algorithms.actor.rynnvla_action_hidden_actor import (
        RynnVLAActionHiddenActor as SplitRynnVLAActionHiddenActor,
    )
    from dreamervla.algorithms.actor.vla_action_head_actor import (
        VLAActionHeadActor as SplitVLAActionHeadActor,
    )
    from dreamervla.algorithms.actor.vla_policy import VLAPolicy as SplitVLAPolicy

    assert LatentToActionHiddenActor is SplitLatentToActionHiddenActor
    assert LatentToOpenVLADiscreteTokenActor is SplitLatentToOpenVLADiscreteTokenActor
    assert LatentToOpenVLAHiddenStateActor is SplitLatentToOpenVLAHiddenStateActor
    assert OpenVLADiscreteTokenActor is SplitOpenVLADiscreteTokenActor
    assert RynnVLAActionHiddenActor is SplitRynnVLAActionHiddenActor
    assert VLAActionHeadActor is SplitVLAActionHeadActor
    assert VLAPolicy is SplitVLAPolicy


def test_latent_to_action_hidden_actor_docs_use_role_based_wm_wording() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "algorithms"
        / "actor"
        / "latent_to_action_hidden_actor.py"
    ).read_text(encoding="utf-8")
    assert ("DINO" + "-WM") not in source
    assert ("dino" + "_wm") not in source.lower()
    assert ("dino" + "wm") not in source.lower()
