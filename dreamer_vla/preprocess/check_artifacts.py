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
) -> None:
    path = _project_path(path)
    files = _hdf5_files(path)
    if require_config and not (path / "preprocess_config.json").is_file():
        raise RuntimeError(f"missing preprocess_config.json under: {path}")

    if reference_dir is not None:
        reference = _project_path(reference_dir)
        reference_names = {item.name for item in _hdf5_files(reference)}
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

    required_demo_datasets = required_demo_datasets or []
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as handle:
                if require_complete_attr and not bool(handle.attrs.get("complete", False)):
                    raise RuntimeError(f"missing complete=true attr: {file_path}")
                data_group = handle.get("data")
                if data_group is None or not data_group.keys():
                    raise RuntimeError(f"missing non-empty data group: {file_path}")
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
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
