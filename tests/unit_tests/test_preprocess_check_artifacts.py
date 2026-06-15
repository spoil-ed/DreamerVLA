from __future__ import annotations

import json

import h5py
import pytest

from dreamervla.preprocess.check_artifacts import (
    validate_hdf5_dir,
    validate_metainfo,
)


def _write_hdf5(path, *, complete: bool | None = None, dataset: str = "obs_embedding") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        if complete is not None:
            handle.attrs["complete"] = complete
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset(dataset, data=[1.0])


def test_validate_metainfo_requires_existing_nonempty_json_object(tmp_path) -> None:
    missing = tmp_path / "metainfo.json"
    with pytest.raises(RuntimeError, match="missing metainfo"):
        validate_metainfo(missing)

    missing.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty object"):
        validate_metainfo(missing)

    missing.write_text(json.dumps({"task": {"demo_0": {"success": True}}}), encoding="utf-8")
    validate_metainfo(missing)


def test_validate_hdf5_dir_matches_reference_names_and_complete_attr(tmp_path) -> None:
    reference = tmp_path / "reward"
    sidecar = tmp_path / "hidden"
    _write_hdf5(reference / "a.hdf5")
    _write_hdf5(reference / "b.hdf5")
    _write_hdf5(sidecar / "a.hdf5", complete=True, dataset="action_hidden_states")

    with pytest.raises(RuntimeError, match="file set mismatch"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["action_hidden_states"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=False, dataset="action_hidden_states")
    with pytest.raises(RuntimeError, match="complete=true"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["action_hidden_states"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=True, dataset="obs_embedding")
    with pytest.raises(RuntimeError, match="missing datasets"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["action_hidden_states"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=True, dataset="action_hidden_states")
    validate_hdf5_dir(
        sidecar,
        reference_dir=reference,
        require_complete_attr=True,
        required_demo_datasets=["action_hidden_states"],
    )


def test_validate_hdf5_dir_can_require_preprocess_config(tmp_path) -> None:
    sidecar = tmp_path / "hidden"
    _write_hdf5(sidecar / "a.hdf5", complete=True)

    with pytest.raises(RuntimeError, match="preprocess_config"):
        validate_hdf5_dir(sidecar, require_config=True)

    (sidecar / "preprocess_config.json").write_text("{}", encoding="utf-8")
    validate_hdf5_dir(sidecar, require_config=True)
