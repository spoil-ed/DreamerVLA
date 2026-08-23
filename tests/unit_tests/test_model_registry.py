from __future__ import annotations

from omegaconf import OmegaConf


def test_model_registry_builds_registered_model_type() -> None:
    from dreamervla.models.registry import get_model, register_model

    calls = []

    def builder(cfg, torch_dtype=None):
        calls.append((cfg.hidden_dim, torch_dtype))
        return {"hidden_dim": int(cfg.hidden_dim), "dtype": torch_dtype}

    register_model("unit_test_model", builder, replace=True)

    model = get_model(
        OmegaConf.create({"model_type": "unit_test_model", "hidden_dim": 4}), torch_dtype="bf16"
    )

    assert model == {"hidden_dim": 4, "dtype": "bf16"}
    assert calls == [(4, "bf16")]


def test_model_registry_reports_supported_types() -> None:
    from dreamervla.models.registry import registered_model_types

    assert "openvla" in registered_model_types()
    assert "openvla_oft" in registered_model_types()


def test_model_registry_rejects_unknown_model_type() -> None:
    import pytest

    from dreamervla.models.registry import get_model

    with pytest.raises(ValueError, match="unknown model_type"):
        get_model(OmegaConf.create({"model_type": "missing_model"}))
