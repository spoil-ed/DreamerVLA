from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dreamervla.utils.component_checkpoint import (
    load_component_checkpoint,
    state_dict_sha256,
)
from dreamervla.utils.hf_module import save_module_pretrained


def test_load_component_checkpoint_supports_wm_and_classifier_schemas(
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

    loaded_wm = load_component_checkpoint(wm_path, "world_model")
    loaded_classifier = load_component_checkpoint(cls_path, "classifier")

    assert loaded_wm.state_dict.keys() == world_model.state_dict().keys()
    assert loaded_classifier.state_dict.keys() == classifier.state_dict().keys()
    assert loaded_classifier.metadata["threshold"] == 0.7
    assert loaded_classifier.metadata["f1"] == 0.9


def test_load_component_checkpoint_supports_runner_state_dict_schema(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "runner.ckpt"
    torch.save({"state_dicts": {"world_model": model.state_dict()}}, path)

    loaded = load_component_checkpoint(path, "world_model")

    assert loaded.state_dict.keys() == model.state_dict().keys()


def test_load_component_checkpoint_rejects_missing_component(tmp_path: Path) -> None:
    path = tmp_path / "bad.ckpt"
    torch.save({"unrelated": {}}, path)

    with pytest.raises(ValueError, match="world_model"):
        load_component_checkpoint(path, "world_model")


def test_state_dict_sha256_is_deterministic_and_state_sensitive() -> None:
    model = nn.Linear(2, 1)
    before = state_dict_sha256(model.state_dict())

    assert state_dict_sha256(model.state_dict()) == before
    with torch.no_grad():
        model.weight.add_(1.0)
    assert state_dict_sha256(model.state_dict()) != before


def test_load_component_checkpoint_supports_huggingface_directory(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 3)
    checkpoint_dir = tmp_path / "world_model_hf"
    save_module_pretrained(
        model,
        str(checkpoint_dir),
        target="torch.nn.Linear",
        init_args={"in_features": 2, "out_features": 3},
    )

    loaded = load_component_checkpoint(checkpoint_dir, "world_model")

    assert loaded.state_dict.keys() == model.state_dict().keys()
    assert loaded.metadata["checkpoint_format"] == "huggingface"
