from __future__ import annotations

import inspect

import pytest

import dreamer_vla.models.world_model as world_model
from dreamer_vla.models.encoder.base_encoder import BaseEncoder
from dreamer_vla.models.world_model import dreamerv3_torch, tssm_torch
from dreamer_vla.models.world_model.base_world_model import (
    BaseWorldModel,
    DreamerV3LatentState,
    DreamerV3Loss,
)
from dreamer_vla.models.world_model.dreamer_v3_pixel_rynn_backbone_world_model import (
    DreamerV3PixelRynnBackboneWorldModel,
)
from dreamer_vla.models.world_model.dreamer_v3_pixel_world_model import DreamerV3PixelWorldModel
from dreamer_vla.models.world_model.dreamer_v3_token_from_pixel_world_model import (
    DreamerV3TokenFromPixelWorldModel,
)
from dreamer_vla.models.world_model.dreamer_v3_token_world_model import DreamerV3TokenWorldModel
from dreamer_vla.models.world_model.tssm_rynn_backbone_world_model import (
    TSSMRynnBackboneWorldModel,
)
from dreamer_vla.models.world_model.tssm_token_rynn_backbone_world_model import (
    TSSMTokenRynnBackboneWorldModel,
)


def test_split_world_model_modules_export_classes() -> None:
    assert issubclass(DreamerV3PixelWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3TokenWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3TokenFromPixelWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3PixelRynnBackboneWorldModel, BaseWorldModel)
    assert issubclass(TSSMRynnBackboneWorldModel, BaseWorldModel)
    assert issubclass(TSSMTokenRynnBackboneWorldModel, BaseWorldModel)
    assert DreamerV3LatentState.__name__ == "DreamerV3LatentState"
    assert DreamerV3Loss.__name__ == "DreamerV3Loss"


def test_model_base_interfaces_are_abstract() -> None:
    assert inspect.isabstract(BaseEncoder)
    assert inspect.isabstract(BaseWorldModel)
    with pytest.raises(TypeError):
        BaseEncoder()
    with pytest.raises(TypeError):
        BaseWorldModel()


def test_world_model_package_exports_split_world_model_classes() -> None:
    assert world_model.DreamerV3PixelWorldModel is DreamerV3PixelWorldModel
    assert world_model.DreamerV3TokenWorldModel is DreamerV3TokenWorldModel
    assert world_model.DreamerV3TokenFromPixelWorldModel is DreamerV3TokenFromPixelWorldModel
    assert (
        world_model.DreamerV3PixelRynnBackboneWorldModel
        is DreamerV3PixelRynnBackboneWorldModel
    )
    assert world_model.TSSMRynnBackboneWorldModel is TSSMRynnBackboneWorldModel
    assert world_model.TSSMTokenRynnBackboneWorldModel is TSSMTokenRynnBackboneWorldModel


def test_foundation_modules_do_not_reexport_route_classes() -> None:
    assert not hasattr(dreamerv3_torch, "DreamerV3PixelWorldModel")
    assert not hasattr(dreamerv3_torch, "DreamerV3TokenWorldModel")
    assert not hasattr(dreamerv3_torch, "DreamerV3TokenFromPixelWorldModel")
    assert not hasattr(dreamerv3_torch, "DreamerV3PixelRynnBackboneWorldModel")
    assert not hasattr(tssm_torch, "TSSMRynnBackboneWorldModel")
    assert not hasattr(tssm_torch, "TSSMTokenRynnBackboneWorldModel")
