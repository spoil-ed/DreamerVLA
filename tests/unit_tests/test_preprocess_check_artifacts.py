from __future__ import annotations

import json

import h5py
import pytest

from dreamervla.preprocess.artifact_utils import (
    Hdf5PreprocessTask,
    assign_tasks_by_frames,
    plan_hdf5_preprocess_tasks,
)
from dreamervla.preprocess.check_artifacts import (
    validate_hdf5_dir,
    validate_metainfo,
)


def _write_hdf5(
    path,
    *,
    complete: bool | None = None,
    dataset: str = "obs_embedding",
    length: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        if complete is not None:
            handle.attrs["complete"] = complete
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset(dataset, data=[1.0] * length)


def _write_source_hdf5(path, *, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=[[0.0]] * frames)


def _canonical_config(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "action_head_type": "oft_discrete_token",
        "obs_hidden_source": "hidden_token",
        "hidden_key": "obs_embedding",
        "token_count": 256,
        "token_dim": 4096,
        "hidden_dim": 1_048_576,
        "obs_embedding_shape": [256, 4096],
        "hidden_storage_format": "tokenized",
        "num_images_in_input": 1,
        "patches_per_image": 256,
        "history": 1,
        "include_state": False,
        "sidecar_schema_version": 1,
        "required_demo_datasets": ["obs_embedding"],
    }
    config.update(overrides)
    return config


def _write_hidden_token_sidecar(path, *, length: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = True
        demo = handle.create_group("data/demo_0")
        demo.create_dataset(
            "obs_embedding",
            shape=(length, 256, 4096),
            dtype="float16",
            chunks=(1, 256, 4096),
        )


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
    _write_hdf5(sidecar / "a.hdf5", complete=True, dataset="required_tensor")

    with pytest.raises(RuntimeError, match="file set mismatch"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["required_tensor"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=False, dataset="required_tensor")
    with pytest.raises(RuntimeError, match="complete=true"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["required_tensor"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=True, dataset="obs_embedding")
    with pytest.raises(RuntimeError, match="missing datasets"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["required_tensor"],
        )

    _write_hdf5(sidecar / "b.hdf5", complete=True, dataset="required_tensor")
    validate_hdf5_dir(
        sidecar,
        reference_dir=reference,
        require_complete_attr=True,
        required_demo_datasets=["required_tensor"],
    )


def test_validate_hdf5_dir_can_match_reference_demo_lengths(tmp_path) -> None:
    reference = tmp_path / "reward"
    sidecar = tmp_path / "hidden"
    _write_source_hdf5(reference / "a.hdf5", frames=3)
    _write_hdf5(
        sidecar / "a.hdf5",
        complete=True,
        dataset="required_tensor",
        length=2,
    )

    with pytest.raises(RuntimeError, match="length mismatch"):
        validate_hdf5_dir(
            sidecar,
            reference_dir=reference,
            require_complete_attr=True,
            required_demo_datasets=["required_tensor"],
            match_reference_demos=True,
            match_reference_lengths=True,
        )

    _write_hdf5(
        sidecar / "a.hdf5",
        complete=True,
        dataset="required_tensor",
        length=3,
    )
    validate_hdf5_dir(
        sidecar,
        reference_dir=reference,
        require_complete_attr=True,
        required_demo_datasets=["required_tensor"],
        match_reference_demos=True,
        match_reference_lengths=True,
    )


def test_validate_hdf5_dir_can_require_preprocess_config(tmp_path) -> None:
    sidecar = tmp_path / "hidden"
    _write_hidden_token_sidecar(sidecar / "a.hdf5")

    with pytest.raises(RuntimeError, match="preprocess_config"):
        validate_hdf5_dir(sidecar, require_config=True)

    (sidecar / "preprocess_config.json").write_text(
        json.dumps(_canonical_config()),
        encoding="utf-8",
    )
    validate_hdf5_dir(sidecar, require_config=True)


def test_validate_hdf5_dir_rejects_extra_required_dataset_alias(tmp_path) -> None:
    sidecar = tmp_path / "hidden"
    _write_hidden_token_sidecar(sidecar / "a.hdf5")
    (sidecar / "preprocess_config.json").write_text(
        json.dumps(_canonical_config(required_demo_datasets=["obs_embedding", "policy_slots"])),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="required_demo_datasets"):
        validate_hdf5_dir(sidecar, require_complete_attr=True, require_config=True)


def test_validate_hdf5_dir_rejects_removed_sidecar_flags(tmp_path) -> None:
    sidecar = tmp_path / "hidden"
    _write_hidden_token_sidecar(sidecar / "a.hdf5")
    (sidecar / "preprocess_config.json").write_text(
        json.dumps(_canonical_config(save_hidden_token=True)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="removed sidecar fields"):
        validate_hdf5_dir(sidecar, require_complete_attr=True, require_config=True)


def test_validate_hdf5_dir_rejects_custom_observation_key(tmp_path) -> None:
    sidecar = tmp_path / "hidden"
    _write_hidden_token_sidecar(sidecar / "a.hdf5")
    (sidecar / "preprocess_config.json").write_text(
        json.dumps(_canonical_config(hidden_key="latent")),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hidden_key='latent'"):
        validate_hdf5_dir(sidecar, require_complete_attr=True, require_config=True)


def test_preprocess_task_plan_filters_complete_and_repairs_partial_outputs(tmp_path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source_hdf5(source / "a.hdf5", frames=5)
    _write_source_hdf5(source / "b.hdf5", frames=7)
    _write_hdf5(out / "a.hdf5", complete=True, dataset="required_tensor", length=5)
    _write_hdf5(out / "b.hdf5", complete=False, dataset="required_tensor")

    plan = plan_hdf5_preprocess_tasks(
        sorted(source.glob("*.hdf5")),
        rank=0,
        world_size=2,
        output_paths=lambda path: [out / path.name],
        required_demo_datasets={
            out / "a.hdf5": ["required_tensor"],
            out / "b.hdf5": ["required_tensor"],
        },
    )

    assert [task.source_path.name for task in plan.skipped] == ["a.hdf5"]
    assert [task.source_path.name for task in plan.repaired] == ["b.hdf5"]
    assert [task.source_path.name for task in plan.pending] == ["b.hdf5"]
    assert not (out / "b.hdf5").exists()
    assert plan.loads_by_rank == [7, 0]


def test_preprocess_task_plan_can_fail_fast_on_partial_outputs(tmp_path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source_hdf5(source / "a.hdf5", frames=5)
    _write_hdf5(out / "a.hdf5", complete=False, dataset="required_tensor")

    with pytest.raises(RuntimeError, match="incomplete preprocessing artifact"):
        plan_hdf5_preprocess_tasks(
            sorted(source.glob("*.hdf5")),
            rank=0,
            world_size=1,
            output_paths=lambda path: [out / path.name],
            required_demo_datasets={out / "a.hdf5": ["required_tensor"]},
            repair=False,
        )


def test_preprocess_task_plan_filters_complete_and_assigns_missing_outputs(tmp_path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source_hdf5(source / "a.hdf5", frames=5)
    _write_source_hdf5(source / "b.hdf5", frames=7)
    _write_hdf5(out / "a.hdf5", complete=True, dataset="required_tensor", length=5)

    plan = plan_hdf5_preprocess_tasks(
        sorted(source.glob("*.hdf5")),
        rank=0,
        world_size=2,
        output_paths=lambda path: [out / path.name],
        required_demo_datasets={out / "a.hdf5": ["required_tensor"]},
    )

    assert [task.source_path.name for task in plan.skipped] == ["a.hdf5"]
    assert [task.source_path.name for task in plan.pending] == ["b.hdf5"]
    assert plan.loads_by_rank == [7, 0]


def test_preprocess_task_plan_uses_fixed_dataset_from_config(tmp_path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source_hdf5(source / "a.hdf5", frames=5)
    _write_hidden_token_sidecar(out / "a.hdf5", length=5)
    (out / "preprocess_config.json").write_text(
        json.dumps(_canonical_config()),
        encoding="utf-8",
    )

    plan = plan_hdf5_preprocess_tasks(
        sorted(source.glob("*.hdf5")),
        rank=0,
        world_size=1,
        output_paths=lambda path: [out / path.name],
    )

    assert [task.source_path.name for task in plan.skipped] == ["a.hdf5"]
    assert plan.pending == []


def test_assign_tasks_by_frames_balances_large_files_first(tmp_path) -> None:
    tasks = [
        Hdf5PreprocessTask(tmp_path / "a.hdf5", demos=1, frames=10),
        Hdf5PreprocessTask(tmp_path / "b.hdf5", demos=1, frames=9),
        Hdf5PreprocessTask(tmp_path / "c.hdf5", demos=1, frames=1),
    ]

    assignments = assign_tasks_by_frames(tasks, world_size=2)
    loads = [sum(task.frames for task in bucket) for bucket in assignments]

    assert sorted(loads) == [10, 10]
