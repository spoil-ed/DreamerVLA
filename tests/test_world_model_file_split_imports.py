from __future__ import annotations

from src.models.world_model import dreamerv3_torch, tssm_torch
from src.models.world_model.base_world_model import (
    BaseWorldModel,
    DreamerV3LatentState,
    DreamerV3Loss,
)
from src.models.world_model.dreamer_v3_pixel_rynn_backbone_world_model import (
    DreamerV3PixelRynnBackboneWorldModel,
)
from src.models.world_model.dreamer_v3_pixel_world_model import DreamerV3PixelWorldModel
from src.models.world_model.dreamer_v3_token_from_pixel_world_model import (
    DreamerV3TokenFromPixelWorldModel,
)
from src.models.world_model.dreamer_v3_token_world_model import DreamerV3TokenWorldModel
from src.models.world_model.tssm_rynn_backbone_world_model import TSSMRynnBackboneWorldModel
from src.models.world_model.tssm_token_rynn_backbone_world_model import (
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


def test_legacy_modules_reexport_split_world_model_classes() -> None:
    assert dreamerv3_torch.DreamerV3PixelWorldModel is DreamerV3PixelWorldModel
    assert dreamerv3_torch.DreamerV3TokenWorldModel is DreamerV3TokenWorldModel
    assert dreamerv3_torch.DreamerV3TokenFromPixelWorldModel is DreamerV3TokenFromPixelWorldModel
    assert dreamerv3_torch.DreamerV3PixelRynnBackboneWorldModel is DreamerV3PixelRynnBackboneWorldModel
    assert tssm_torch.TSSMRynnBackboneWorldModel is TSSMRynnBackboneWorldModel
    assert tssm_torch.TSSMTokenRynnBackboneWorldModel is TSSMTokenRynnBackboneWorldModel
