from __future__ import annotations

import subprocess
import sys
import textwrap

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

    assert "openvla_oft" in registered_model_types()


def test_removed_openvla_implementation_is_rejected_before_construction() -> None:
    import pytest

    from dreamervla.models.registry import get_model, validate_model_type

    with pytest.raises(ValueError, match="unknown model_type"):
        validate_model_type("openvla")
    with pytest.raises(ValueError, match="unknown model_type"):
        get_model(OmegaConf.create({"model_type": "openvla"}))


def test_model_registry_rejects_unknown_model_type() -> None:
    import pytest

    from dreamervla.models.registry import get_model

    with pytest.raises(ValueError, match="unknown model_type"):
        get_model(OmegaConf.create({"model_type": "missing_model"}))


def test_model_registry_does_not_require_model_runtime_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            textwrap.dedent("""
                import importlib.abc
                import sys

                class BlockModelDependencies(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split('.')[0] in {'torch', 'transformers', 'openpi', 'ray'}:
                            raise ImportError(f'Unexpected model dependency: {fullname}')

                sys.meta_path.insert(0, BlockModelDependencies())
                from dreamervla.models import registered_model_types, validate_model_type
                import dreamervla.models.embodiment

                validate_model_type('openvla_oft')
                assert 'openvla_oft' in registered_model_types()
            """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_world_model_export_does_not_require_vla_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            textwrap.dedent("""
                import importlib.abc
                import sys

                class BlockVLADependencies(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split('.')[0] in {'transformers', 'openpi'}:
                            raise ImportError(f'Unexpected VLA dependency: {fullname}')

                sys.meta_path.insert(0, BlockVLADependencies())
                from dreamervla.models import WorldModel
                from dreamervla.models.embodiment import WorldModel as EmbodimentWorldModel
                from dreamervla.models.embodiment.world_model.wm import WorldModel as Implementation

                assert WorldModel is EmbodimentWorldModel is Implementation
            """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
