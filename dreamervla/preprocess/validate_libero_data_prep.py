"""Validate the LIBERO preprocessing artifact tree.

This checker is intentionally shallow and fast: it verifies that each stage
created the expected files and that stage-4 counts line up, without opening the
tokenized pkl payloads.  Use ``validate_pretokenized.py`` for deep pkl content
checks after this structural audit passes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dreamervla.preprocess.paths import PROJECT_ROOT
from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.paths import data_root as default_data_root
from dreamervla.utils.paths import processed_data_path

STANDARD_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
SPLITS = ("train", "val_ind", "val_ood")


@dataclass(frozen=True)
class ValidationIssue:
    """One concrete validation failure."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class LiberoDataPrepSpec:
    """Expected preprocessing layout for one LIBERO suite."""

    suite: str
    data_root: Path
    processed_data_root: Path
    his: int = 1
    action_horizon: int = 1
    image_resolution: int = 256
    check_configs: bool = True
    check_action_hidden: bool = False

    @property
    def suffix(self) -> str:
        return (
            f"his_{self.his}_third_view_wrist_w_state_"
            f"{self.action_horizon}_{self.image_resolution}"
        )

    @property
    def hdf5_dir(self) -> Path:
        return self.processed_data_root / f"no_noops_t_{self.image_resolution}"

    @property
    def reward_dir(self) -> Path:
        return Path(f"{self.hdf5_dir}_remaining_reward")

    @property
    def image_state_dir(self) -> Path:
        return self.processed_data_root / f"image_state_action_t_{self.image_resolution}"

    @property
    def hidden_dir(self) -> Path:
        return Path(f"{self.hdf5_dir}_legacy_action_hidden_vla_policy_h2")

    @property
    def legacy_prefixed_stage_dirs(self) -> tuple[Path, ...]:
        """Old stage layout that repeated the artifact name inside its root."""

        return (
            self.processed_data_root
            / f"{self.suite}_no_noops_t_{self.image_resolution}",
            self.processed_data_root
            / f"{self.suite}_no_noops_t_{self.image_resolution}_remaining_reward",
            self.processed_data_root
            / f"{self.suite}_image_state_action_t_{self.image_resolution}",
        )

    def conv_path(self, split: str) -> Path:
        return (
            self.processed_data_root
            / "convs"
            / f"{self.suite}_his_{self.his}_{split}_third_view_wrist_w_state_"
            f"{self.action_horizon}_{self.image_resolution}.json"
        )

    def token_dir(self, split: str) -> Path:
        return (
            self.processed_data_root
            / "tokens"
            / f"{self.suite}_his_{self.his}_{split}_third_view_wrist_w_state_"
            f"{self.action_horizon}_{self.image_resolution}"
        )

    @property
    def manifest_path(self) -> Path:
        return (
            self.processed_data_root
            / "concate_tokens"
            / f"{self.suite}_{self.suffix}.json"
        )

    @property
    def config_dir(self) -> Path:
        return self.data_root / "configs" / self.suite


@dataclass
class SuiteValidationReport:
    """Validation result for one suite."""

    suite: str
    summary: dict[str, int] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def _count_hdf5(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.hdf5") if item.is_file())


def _count_child_dirs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_dir())


def _count_pkls(path: Path) -> int:
    files_dir = path / "files"
    if not files_dir.is_dir():
        return 0
    return sum(1 for item in files_dir.glob("*.pkl") if item.is_file())


def _load_json_list(
    path: Path,
    issues: list[ValidationIssue],
    *,
    missing_code: str,
    invalid_code: str,
    not_list_code: str,
) -> list[Any] | None:
    if not path.is_file():
        issues.append(
            ValidationIssue(
                missing_code,
                str(path),
                f"missing JSON list: {path}",
            )
        )
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(
            ValidationIssue(
                invalid_code,
                str(path),
                f"invalid JSON in {path}: {exc}",
            )
        )
        return None

    if not isinstance(data, list):
        issues.append(
            ValidationIssue(
                not_list_code,
                str(path),
                f"expected a JSON list: {path}",
            )
        )
        return None

    return data


def _resolve_record_path(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    base_candidate = (base_dir / candidate).resolve()
    if base_candidate.exists():
        return base_candidate

    return (PROJECT_ROOT / candidate).resolve()


def _record_file_missing_count(records: list[Any], base_dir: Path) -> int:
    missing = 0
    for record in records:
        if not isinstance(record, dict):
            missing += 1
            continue
        file_value = record.get("file")
        if not isinstance(file_value, str) or not file_value:
            missing += 1
            continue
        if not _resolve_record_path(file_value, base_dir).is_file():
            missing += 1
    return missing


def _extract_meta_path(config_path: Path) -> str | None:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if not stripped.startswith("path:"):
            continue
        return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _add_count_mismatch(
    issues: list[ValidationIssue],
    *,
    code: str,
    path: Path,
    left_label: str,
    left_count: int,
    right_label: str,
    right_count: int,
) -> None:
    if left_count == right_count:
        return
    issues.append(
        ValidationIssue(
            code,
            str(path),
            (
                f"{left_label} count {left_count} does not match "
                f"{right_label} count {right_count}: {path}"
            ),
        )
    )


def _validate_configs(
    spec: LiberoDataPrepSpec,
    issues: list[ValidationIssue],
) -> None:
    expected_configs = {
        f"{spec.suffix}_pretokenize.yaml": spec.manifest_path,
        f"{spec.suffix}_pretokenize_val_ind.yaml": spec.token_dir("val_ind")
        / "record.json",
        f"{spec.suffix}_pretokenize_val_ood.yaml": spec.token_dir("val_ood")
        / "record.json",
    }

    for filename, expected_target in expected_configs.items():
        config_path = spec.config_dir / filename
        if not config_path.is_file():
            issues.append(
                ValidationIssue(
                    "config_missing",
                    str(config_path),
                    f"missing generated dataset config: {config_path}",
                )
            )
            continue

        meta_path = _extract_meta_path(config_path)
        if meta_path is None:
            issues.append(
                ValidationIssue(
                    "config_invalid",
                    str(config_path),
                    f"config has no META path entry: {config_path}",
                )
            )
            continue

        resolved = _resolve_record_path(meta_path, config_path.parent)
        if not resolved.is_file():
            issues.append(
                ValidationIssue(
                    "config_path_missing",
                    str(config_path),
                    f"config META path does not exist: {meta_path}",
                )
            )
            continue

        if expected_target.is_file() and resolved != expected_target.resolve():
            issues.append(
                ValidationIssue(
                    "config_path_mismatch",
                    str(config_path),
                    (
                        f"config META path resolves to {resolved}, expected "
                        f"{expected_target.resolve()}"
                    ),
                )
            )


def validate_suite(spec: LiberoDataPrepSpec) -> SuiteValidationReport:
    """Validate one LIBERO suite and return a structured report."""

    issues: list[ValidationIssue] = []
    summary: dict[str, int] = {}

    if not spec.processed_data_root.is_dir():
        issues.append(
            ValidationIssue(
                "artifact_root_missing",
                str(spec.processed_data_root),
                f"preprocessing artifact root does not exist: {spec.processed_data_root}",
            )
        )
    for legacy_dir in spec.legacy_prefixed_stage_dirs:
        if legacy_dir.exists():
            issues.append(
                ValidationIssue(
                    "legacy_prefixed_stage_dir",
                    str(legacy_dir),
                    (
                        "found old repeated-prefix stage directory; expected "
                        f"stage directories directly under {spec.processed_data_root}: "
                        f"{legacy_dir}"
                    ),
                )
            )

    raw_hdf5 = _count_hdf5(spec.hdf5_dir)
    reward_hdf5 = _count_hdf5(spec.reward_dir)
    image_dirs = _count_child_dirs(spec.image_state_dir)
    summary["raw_hdf5"] = raw_hdf5
    summary["reward_hdf5"] = reward_hdf5
    summary["image_tree_dirs"] = image_dirs

    if raw_hdf5 == 0:
        issues.append(
            ValidationIssue(
                "raw_hdf5_missing",
                str(spec.hdf5_dir),
                f"no HDF5 files found in final no-noops dir: {spec.hdf5_dir}",
            )
        )
    if reward_hdf5 == 0:
        issues.append(
            ValidationIssue(
                "reward_hdf5_missing",
                str(spec.reward_dir),
                f"no HDF5 files found in remaining-reward dir: {spec.reward_dir}",
            )
        )
    elif raw_hdf5 > 0:
        _add_count_mismatch(
            issues,
            code="reward_count_mismatch",
            path=spec.reward_dir,
            left_label="reward HDF5",
            left_count=reward_hdf5,
            right_label="raw HDF5",
            right_count=raw_hdf5,
        )

    if image_dirs == 0:
        issues.append(
            ValidationIssue(
                "image_tree_missing",
                str(spec.image_state_dir),
                f"no task image/state/action directories found: {spec.image_state_dir}",
            )
        )
    elif raw_hdf5 > 0 and image_dirs < raw_hdf5:
        issues.append(
            ValidationIssue(
                "image_tree_incomplete",
                str(spec.image_state_dir),
                (
                    f"image/state/action task dirs {image_dirs} are fewer than "
                    f"raw HDF5 files {raw_hdf5}: {spec.image_state_dir}"
                ),
            )
        )

    split_record_counts: dict[str, int] = {}
    for split in SPLITS:
        conv_path = spec.conv_path(split)
        conv_records = _load_json_list(
            conv_path,
            issues,
            missing_code="conv_missing",
            invalid_code="conv_invalid",
            not_list_code="conv_not_list",
        )
        conv_count = len(conv_records) if conv_records is not None else 0
        summary[f"conv_{split}"] = conv_count
        if conv_records is not None and conv_count == 0:
            issues.append(
                ValidationIssue(
                    "conv_empty",
                    str(conv_path),
                    f"conversation split is empty: {conv_path}",
                )
            )

        token_dir = spec.token_dir(split)
        token_count = _count_pkls(token_dir)
        summary[f"token_{split}"] = token_count
        _add_count_mismatch(
            issues,
            code="token_count_mismatch",
            path=token_dir,
            left_label="token pkl",
            left_count=token_count,
            right_label="conv",
            right_count=conv_count,
        )

        record_path = token_dir / "record.json"
        record_records = _load_json_list(
            record_path,
            issues,
            missing_code="record_missing",
            invalid_code="record_invalid",
            not_list_code="record_not_list",
        )
        record_count = len(record_records) if record_records is not None else 0
        split_record_counts[split] = record_count
        summary[f"record_{split}"] = record_count
        if record_records is not None:
            _add_count_mismatch(
                issues,
                code="record_count_mismatch",
                path=record_path,
                left_label="record",
                left_count=record_count,
                right_label="token pkl",
                right_count=token_count,
            )
            missing_files = _record_file_missing_count(
                record_records, record_path.parent
            )
            if missing_files:
                issues.append(
                    ValidationIssue(
                        "record_file_missing",
                        str(record_path),
                        (
                            f"{missing_files} record entries point to missing "
                            f"pkl files: {record_path}"
                        ),
                    )
                )

    manifest_records = _load_json_list(
        spec.manifest_path,
        issues,
        missing_code="manifest_missing",
        invalid_code="manifest_invalid",
        not_list_code="manifest_not_list",
    )
    manifest_count = len(manifest_records) if manifest_records is not None else 0
    summary["manifest"] = manifest_count
    expected_manifest_count = sum(split_record_counts.values())
    if manifest_records is not None:
        _add_count_mismatch(
            issues,
            code="manifest_count_mismatch",
            path=spec.manifest_path,
            left_label="manifest",
            left_count=manifest_count,
            right_label="record",
            right_count=expected_manifest_count,
        )
        missing_files = _record_file_missing_count(
            manifest_records, spec.manifest_path.parent
        )
        if missing_files:
            issues.append(
                ValidationIssue(
                    "manifest_file_missing",
                    str(spec.manifest_path),
                    (
                        f"{missing_files} manifest entries point to missing "
                        f"pkl files: {spec.manifest_path}"
                    ),
                )
            )

    if spec.check_configs:
        _validate_configs(spec, issues)

    if spec.check_action_hidden:
        hidden_hdf5 = _count_hdf5(spec.hidden_dir)
        summary["hidden_hdf5"] = hidden_hdf5
        if hidden_hdf5 == 0:
            issues.append(
                ValidationIssue(
                    "hidden_hdf5_missing",
                    str(spec.hidden_dir),
                    f"no HDF5 files found in action-hidden dir: {spec.hidden_dir}",
                )
            )
        elif raw_hdf5 > 0:
            _add_count_mismatch(
                issues,
                code="hidden_count_mismatch",
                path=spec.hidden_dir,
                left_label="hidden HDF5",
                left_count=hidden_hdf5,
                right_label="raw HDF5",
                right_count=raw_hdf5,
            )

    return SuiteValidationReport(spec.suite, summary=summary, issues=issues)


def _print_report(report: SuiteValidationReport) -> None:
    status = "OK" if report.ok else "FAIL"
    print(f"── {report.suite}: {status} ──")
    print(
        "  hdf5 raw={raw_hdf5} reward={reward_hdf5} image_tree={image_tree_dirs}".format(
            **report.summary
        )
    )
    for split in SPLITS:
        print(
            "  {split:<7s} conv={conv:<7d} token_pkl={token:<7d} record={record:<7d}".format(
                split=split,
                conv=report.summary.get(f"conv_{split}", 0),
                token=report.summary.get(f"token_{split}", 0),
                record=report.summary.get(f"record_{split}", 0),
            )
        )
    print(f"  manifest={report.summary.get('manifest', 0)}")
    if "hidden_hdf5" in report.summary:
        print(f"  action_hidden_hdf5={report.summary['hidden_hdf5']}")
    for issue in report.issues:
        print(f"  [{issue.code}] {issue.message}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = script_namespace("validate_libero_data_prep", argv)

    data_root = Path(args.data_root).expanduser()
    processed_root_base = (
        Path(args.processed_data_root).expanduser()
        if args.processed_data_root is not None
        else processed_data_path()
    )
    if args.processed_data_root is None and args.data_root != default_data_root():
        processed_root_base = data_root / "processed_data"

    reports = [
        validate_suite(
            LiberoDataPrepSpec(
                suite=suite,
                data_root=data_root,
                processed_data_root=(
                    processed_root_base
                    if args.processed_data_root is not None
                    else processed_root_base / suite
                ),
                his=args.his,
                action_horizon=args.action_horizon,
                image_resolution=args.image_resolution,
                check_configs=not args.skip_configs,
                check_action_hidden=args.check_action_hidden,
            )
        )
        for suite in args.suites
    ]

    print(
        f"[validate_libero_data_prep] data_root={data_root} "
        f"processed_data_root={processed_root_base}"
    )
    print()
    for report in reports:
        _print_report(report)

    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
