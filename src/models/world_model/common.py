"""Truly model-agnostic primitives shared across DreamerVLA's world models.

This is the equivalent of ``diffusion_policy/model/common/`` — only stuff that
is **not tied to any specific WM architecture** lives here. Things that are
inherently part of a model family (RSSM/DreamerV3 details, TSSM details,
pi0-action-hidden-specific decoders) stay grouped in their family files
(``dreamerv3_torch.py`` and ``tssm_torch.py``).

What's here:

    norms / activation
        ``RMSNorm``, ``ChannelRMSNorm``, ``act`` (activation factory).
    MLP family
        ``MLPHead``, ``ResBlock``, ``ResMLPHead`` — plain and residual MLPs
        used by hidden decoders, reward heads, continue heads, etc.
    Block-diagonal Linear
        ``BlockLinear`` — used inside RSSM's GRU, also reusable elsewhere.
    Tiny utilities
        ``_module_dtype``, ``_module_device``.

What's intentionally NOT here (lives with its family):

    * ``DreamerV3RSSM``, ``DreamerV3PixelEncoder/Decoder``, pi0 hidden
      decoders and RynnBackboneObsEncoder — grouped with their model family;
      reward heads live in ``reward_heads.py``.
    * Causal transformer + multihead attention (ported from TransDreamer,
      tightly coupled with TSSM dynamics) — in ``tssm_torch.py``.

Cfg usage (Hydra ``_target_``):

.. code-block:: yaml

    hidden_decoder:
      _target_: src.models.world_model.common.ResMLPHead
      in_dim: 10240
      out_dim: 35840
      layers: 4
      units: 16384
"""
# norms & activation
from src.models.world_model.dreamerv3_torch import (
    RMSNorm,
    ChannelRMSNorm,
    _act as act,
)
# MLP family
from src.models.world_model.dreamerv3_torch import (
    MLPHead,
    _ResBlock as ResBlock,
    ResMLPHead,
)
# block-diagonal linear
from src.models.world_model.block_linear import BlockLinear
# tiny utils
from src.models.world_model.dreamerv3_torch import _module_dtype, _module_device


__all__ = [
    "RMSNorm", "ChannelRMSNorm", "act",
    "MLPHead", "ResBlock", "ResMLPHead",
    "BlockLinear",
    "_module_dtype", "_module_device",
]
