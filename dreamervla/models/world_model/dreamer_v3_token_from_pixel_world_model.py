from __future__ import annotations

from dreamervla.models.world_model._dreamer_v3_token_common import (
    _DreamerV3TokenWorldModelBase,
)


class DreamerV3TokenFromPixelWorldModel(_DreamerV3TokenWorldModelBase):
    """Pixel-world-model copy with only the observation distribution changed.

    This is the controlled ablation requested for token observations: keep the
    same RSSM, reward head, continue head, KL scales, and loss aggregation as
    ``DreamerV3PixelWorldModel``. The only replacement is:

      pixel obs + MSE decoder loss -> spatial token obs + categorical CE loss.
    """

    _dones_supports_is_terminal = False


__all__ = ["DreamerV3TokenFromPixelWorldModel"]
