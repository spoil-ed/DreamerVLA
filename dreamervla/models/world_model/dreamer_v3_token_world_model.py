from __future__ import annotations

from dreamervla.models.world_model._dreamer_v3_token_common import (
    _DreamerV3TokenWorldModelBase,
)


class DreamerV3TokenWorldModel(_DreamerV3TokenWorldModelBase):
    """DreamerV3 RSSM with categorical image-token observations.

    This is the controlled token counterpart of ``DreamerV3PixelWorldModel``:
    same RSSM, same aggregate free-nats KL semantics, same reward and continue
    heads. The observation likelihood is the only intended change, replacing
    pixel MSE with categorical CE over image tokens.
    """

    _dones_supports_is_terminal = True


__all__ = ["DreamerV3TokenWorldModel"]
