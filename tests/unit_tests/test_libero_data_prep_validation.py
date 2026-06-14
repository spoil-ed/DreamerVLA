from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _touch_pkls(split_dir: Path, count: int) -> None:
    files_dir = split_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (files_dir / f"{idx}.pkl").write_bytes(b"stub")


def _split_stem(suite: str, split: str) -> str:
    return f"{suite}_his_1_{split}_third_view_wrist_w_state_1_256"


def test_libero_data_prep_validation_flags_empty_stage4_outputs(tmp_path: Path) -> None:
    from dreamer_vla.preprocess.validate_libero_data_prep import (
        LiberoDataPrepSpec,
        validate_suite,
    )

    processed = tmp_path / "processed_data"
    data_root = tmp_path
    suite = "libero_goal"

    raw_dir = processed / "no_noops_t_256"
    reward_dir = processed / "no_noops_t_256_remaining_reward"
    image_dir = processed / "image_state_action_t_256"
    for idx in range(2):
        (raw_dir / f"demo_{idx}.hdf5").parent.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"demo_{idx}.hdf5").touch()
        (reward_dir / f"demo_{idx}.hdf5").parent.mkdir(parents=True, exist_ok=True)
        (reward_dir / f"demo_{idx}.hdf5").touch()
        (image_dir / f"task_{idx}").mkdir(parents=True, exist_ok=True)

    _write_json(
        processed / "convs" / f"{_split_stem(suite, 'train')}.json",
        [{"id": 0}, {"id": 1}],
    )
    _write_json(
        processed / "convs" / f"{_split_stem(suite, 'val_ind')}.json",
        [{"id": 2}],
    )
    _write_json(
        processed / "convs" / f"{_split_stem(suite, 'val_ood')}.json",
        [{"id": 3}],
    )

    report = validate_suite(
        LiberoDataPrepSpec(
            suite=suite,
            data_root=data_root,
            processed_data_root=processed,
            his=1,
            action_horizon=1,
            image_resolution=256,
        )
    )

    codes = {issue.code for issue in report.issues}
    assert not report.ok
    assert "token_count_mismatch" in codes
    assert "record_missing" in codes
    assert "manifest_missing" in codes


def test_libero_data_prep_validation_accepts_complete_stage4_outputs(tmp_path: Path) -> None:
    from dreamer_vla.preprocess.validate_libero_data_prep import (
        LiberoDataPrepSpec,
        validate_suite,
    )

    processed = tmp_path / "processed_data"
    data_root = tmp_path
    suite = "libero_goal"
    suffix = "his_1_third_view_wrist_w_state_1_256"
    splits = {"train": 2, "val_ind": 1, "val_ood": 1}

    raw_dir = processed / "no_noops_t_256"
    reward_dir = processed / "no_noops_t_256_remaining_reward"
    image_dir = processed / "image_state_action_t_256"
    for idx in range(2):
        (raw_dir / f"demo_{idx}.hdf5").parent.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"demo_{idx}.hdf5").touch()
        (reward_dir / f"demo_{idx}.hdf5").parent.mkdir(parents=True, exist_ok=True)
        (reward_dir / f"demo_{idx}.hdf5").touch()
        (image_dir / f"task_{idx}").mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for split, count in splits.items():
        split_name = _split_stem(suite, split)
        conv_path = processed / "convs" / f"{split_name}.json"
        token_dir = processed / "tokens" / split_name
        records = []
        for idx in range(count):
            pkl_path = token_dir / "files" / f"{idx}.pkl"
            records.append({"file": str(pkl_path), "id": idx})
        _write_json(conv_path, [{"id": idx} for idx in range(count)])
        _touch_pkls(token_dir, count)
        _write_json(token_dir / "record.json", records)
        manifest_records.extend(records)

    manifest_path = processed / "concate_tokens" / f"{suite}_{suffix}.json"
    _write_json(manifest_path, manifest_records)
    config_dir = data_root / "configs" / suite
    config_dir.mkdir(parents=True)
    for name, path in (
        (f"{suffix}_pretokenize.yaml", manifest_path),
        (
            f"{suffix}_pretokenize_val_ind.yaml",
            processed / "tokens" / _split_stem(suite, "val_ind") / "record.json",
        ),
        (
            f"{suffix}_pretokenize_val_ood.yaml",
            processed / "tokens" / _split_stem(suite, "val_ood") / "record.json",
        ),
    ):
        (config_dir / name).write_text(
            f"META:\n  - path: '{path}'\nprompt_text: 'Finish the task.'\n",
            encoding="utf-8",
        )

    report = validate_suite(
        LiberoDataPrepSpec(
            suite=suite,
            data_root=data_root,
            processed_data_root=processed,
            his=1,
            action_horizon=1,
            image_resolution=256,
        )
    )

    assert report.ok
    assert report.summary["manifest"] == 4
    assert report.issues == []


def test_libero_data_prep_validation_rejects_repeated_prefix_stage_dirs(
    tmp_path: Path,
) -> None:
    from dreamer_vla.preprocess.validate_libero_data_prep import (
        LiberoDataPrepSpec,
        validate_suite,
    )

    processed = tmp_path / "processed_data" / "RynnVLA_LIBERO_libero_goal"
    legacy_stage = "RynnVLA_LIBERO_libero_goal_" + "no_noops_t_256"
    (processed / legacy_stage).mkdir(
        parents=True
    )

    report = validate_suite(
        LiberoDataPrepSpec(
            suite="RynnVLA_LIBERO_libero_goal",
            data_root=tmp_path,
            processed_data_root=processed,
            his=1,
            action_horizon=1,
            image_resolution=256,
            check_configs=False,
        )
    )

    assert not report.ok
    assert "legacy_prefixed_stage_dir" in {issue.code for issue in report.issues}
