from __future__ import annotations

import inspect

import pytest

from dreamervla.models.encoder.base_encoder import BaseEncoder
from dreamervla.models.world_model.base_world_model import BaseWorldModel


def test_model_base_interfaces_are_abstract() -> None:
    assert inspect.isabstract(BaseEncoder)
    assert inspect.isabstract(BaseWorldModel)
    with pytest.raises(TypeError):
        BaseEncoder()
    with pytest.raises(TypeError):
        BaseWorldModel()
