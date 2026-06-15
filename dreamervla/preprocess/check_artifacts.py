"""Fast structural checks for resumable preprocessing stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _hdf5_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise RuntimeError(f"HDF5 directory does not exist: {path}")
    files = sorted(item for item in path.glob("*.hdf5") if item.is_file())
    if not files:
        raise RuntimeError(f"No HDF5 files found under: {path}")
    return files


def _demo_frame_lengths(handle: h5py.File, path: Path) -> dict[str, int]:
    data_group = handle.get("data")
    if data_group is None or not data_group.keys():
        raise RuntimeError(f"missing non-empty data group: {path}")
    lengths: dict[str, int] = {}
    for demo_key, demo in data_group.items():
        if "actions" in demo:
            lengths[str(demo_key)] = int(demo["actions"].shape[0])
            continue
        obs_group = demo.get("obs")
        if obs_group is not None and obs_group.keys():
            first_key = next(iter(obs_group.keys()))
            lengths[str(demo_key)] = int(obs_group[first_key].shape[0])
            continue
        first_dataset = next(
            (value for value in demo.values() if isinstance(value, h5py.Dataset)),
            None,
        )
        if first_dataset is not None and first_dataset.shape:
            lengths[str(demo_key)] = int(first_dataset.shape[0])
            continue
        raise RuntimeError(f"cannot infer demo length for {path}:{demo_key}")
    return lengths


def validate_metainfo(path: str | Path) -> None:
    path = _project_path(path)
    if not path.is_file():
        raise RuntimeError(f"missing metainfo JSON: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid metainfo JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"metainfo JSON must be a non-empty object: {path}")


def validate_hdf5_dir(
    path: str | Path,
    *,
    reference_dir: str | Path | None = None,
    require_complete_attr: bool = False,
    required_demo_datasets: list[str] | None = None,
    require_config: bool = False,
    match_reference_demos: bool = False,
    match_reference_lengths: bool = False,
) -> None:
    path = _project_path(path)
    files = _hdf5_files(path)
    if require_config and not (path / "preprocess_config.json").is_file():
        raise RuntimeError(f"missing preprocess_config.json under: {path}")

    if reference_dir is not None:
        reference = _project_path(reference_dir)
        reference_files = _hdf5_files(reference)
        reference_by_name = {item.name: item for item in reference_files}
        reference_names = set(reference_by_name)
        names = {item.name for item in files}
        missing = sorted(reference_names - names)
        extra = sorted(names - reference_names)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing={missing[:5]}")
            if extra:
                detail.append(f"extra={extra[:5]}")
            raise RuntimeError(
                f"HDF5 file set mismatch for {path} vs {reference}: "
                + ", ".join(detail)
            )
    elif match_reference_demos or match_reference_lengths:
        raise RuntimeError(
            "--match-reference-demos/--match-reference-lengths require --reference-dir"
        )

    required_demo_datasets = required_demo_datasets or []
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as handle:
                if require_complete_attr and not bool(handle.attrs.get("complete", False)):
                    raise RuntimeError(f"missing complete=true attr: {file_path}")
                data_group = handle.get("data")
                if data_group is None or not data_group.keys():
                    raise RuntimeError(f"missing non-empty data group: {file_path}")
                reference_lengths: dict[str, int] | None = None
                if reference_dir is not None and (match_reference_demos or match_reference_lengths):
                    reference_path = reference_by_name[file_path.name]
                    with h5py.File(reference_path, "r") as reference_handle:
                        reference_data = reference_handle.get("data")
                        if reference_data is None or not reference_data.keys():
                            raise RuntimeError(
                                f"missing non-empty reference data group: {reference_path}"
                            )
                        reference_demo_keys = {str(key) for key in reference_data.keys()}
                    demo_keys = {str(key) for key in data_group.keys()}
                    if match_reference_demos and demo_keys != reference_demo_keys:
                        missing_demos = sorted(reference_demo_keys - demo_keys)
                        extra_demos = sorted(demo_keys - reference_demo_keys)
                        detail = []
                        if missing_demos:
                            detail.append(f"missing_demos={missing_demos[:5]}")
                        if extra_demos:
                            detail.append(f"extra_demos={extra_demos[:5]}")
                        raise RuntimeError(
                            f"HDF5 demo set mismatch for {file_path}: "
                            + ", ".join(detail)
                        )
                    if match_reference_lengths:
                        with h5py.File(reference_path, "r") as reference_handle:
                            reference_lengths = _demo_frame_lengths(
                                reference_handle, reference_path
                            )
                if required_demo_datasets:
                    for demo_key in data_group.keys():
                        demo = data_group[demo_key]
                        missing = [
                            key for key in required_demo_datasets if key not in demo
                        ]
                        if missing:
                            raise RuntimeError(
                                f"missing datasets {missing} in {file_path}:{demo_key}"
                            )
                        if reference_lengths is not None:
                            for dataset in required_demo_datasets:
                                value = demo[dataset]
                                if not isinstance(value, h5py.Dataset) or not value.shape:
                                    raise RuntimeError(
                                        f"dataset has no frame dimension in "
                                        f"{file_path}:{demo_key}/{dataset}"
                                    )
                                expected = int(reference_lengths[str(demo_key)])
                                actual = int(value.shape[0])
                                if actual != expected:
                                    raise RuntimeError(
                                        f"dataset length mismatch in "
                                        f"{file_path}:{demo_key}/{dataset}: "
                                        f"actual={actual} expected={expected}"
                                    )
        except OSError as exc:
            raise RuntimeError(f"cannot open HDF5 file {file_path}: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metainfo = subparsers.add_parser("metainfo")
    metainfo.add_argument("--path", required=True)

    hdf5_dir = subparsers.add_parser("hdf5-dir")
    hdf5_dir.add_argument("--dir", required=True)
    hdf5_dir.add_argument("--reference-dir", default=None)
    hdf5_dir.add_argument("--require-complete-attr", action="store_true")
    hdf5_dir.add_argument("--require-config", action="store_true")
    hdf5_dir.add_argument("--match-reference-demos", action="store_true")
    hdf5_dir.add_argument("--match-reference-lengths", action="store_true")
    hdf5_dir.add_argument("--required-demo-dataset", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "metainfo":
        validate_metainfo(args.path)
        return
    if args.command == "hdf5-dir":
        validate_hdf5_dir(
            args.dir,
            reference_dir=args.reference_dir,
            require_complete_attr=bool(args.require_complete_attr),
            required_demo_datasets=list(args.required_demo_dataset),
            require_config=bool(args.require_config),
            match_reference_demos=bool(args.match_reference_demos),
            match_reference_lengths=bool(args.match_reference_lengths),
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
