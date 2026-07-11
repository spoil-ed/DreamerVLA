from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dreamervla.utils.frozen_components import (
    assert_module_frozen,
    load_frozen_component,
    module_state_sha256,
    state_dict_sha256,
)


def test_load_frozen_component_supports_wm_and_classifier_schemas(
    tmp_path: Path,
) -> None:
    world_model = nn.Linear(2, 2)
    classifier = nn.Linear(2, 1)
    wm_path = tmp_path / "wm.ckpt"
    cls_path = tmp_path / "classifier.ckpt"
    torch.save({"world_model": world_model.state_dict()}, wm_path)
    torch.save(
        {
            "model": classifier.state_dict(),
            "threshold": 0.7,
            "f1": 0.9,
        },
        cls_path,
    )

    loaded_wm = load_frozen_component(wm_path, "world_model")
    loaded_classifier = load_frozen_component(cls_path, "classifier")

    assert loaded_wm.state_dict.keys() == world_model.state_dict().keys()
    assert loaded_classifier.state_dict.keys() == classifier.state_dict().keys()
    assert loaded_classifier.metadata["threshold"] == 0.7
    assert loaded_classifier.metadata["f1"] == 0.9


def test_load_frozen_component_supports_runner_state_dict_schema(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "runner.ckpt"
    torch.save({"state_dicts": {"world_model": model.state_dict()}}, path)

    loaded = load_frozen_component(path, "world_model")

    assert loaded.state_dict.keys() == model.state_dict().keys()


def test_load_frozen_component_rejects_missing_component(tmp_path: Path) -> None:
    path = tmp_path / "bad.ckpt"
    torch.save({"unrelated": {}}, path)

    with pytest.raises(ValueError, match="world_model"):
        load_frozen_component(path, "world_model")


def test_module_state_sha256_is_deterministic_and_state_sensitive() -> None:
    model = nn.Linear(2, 1)
    before = module_state_sha256(model)

    assert state_dict_sha256(model.state_dict()) == before
    assert module_state_sha256(model) == before
    with torch.no_grad():
        model.weight.add_(1.0)
    assert module_state_sha256(model) != before


def test_assert_module_frozen_requires_eval_and_no_trainable_parameters() -> None:
    model = nn.Linear(2, 1)

    with pytest.raises(RuntimeError, match="eval mode"):
        assert_module_frozen(model, name="world_model")

    model.eval()
    with pytest.raises(RuntimeError, match="trainable"):
        assert_module_frozen(model, name="world_model")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_module_frozen(model, name="world_model")
