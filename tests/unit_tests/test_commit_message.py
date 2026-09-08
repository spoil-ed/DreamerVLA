from __future__ import annotations

from dreamervla.diagnostics.checks.check_commit_message import validate_subject


def test_commit_subject_accepts_current_dreamervla_style() -> None:
    assert validate_subject("[fix]: restore green baseline") is None
    assert validate_subject("[init]: initial commit") is None


def test_commit_subject_rejects_legacy_conventional_commit_style() -> None:
    error = validate_subject("fix(pi05): restore green baseline")

    assert error is not None
    assert "[type]: description" in error


def test_commit_subject_rejects_unknown_type_and_long_subject() -> None:
    assert validate_subject("[unknown]: change") is not None
    assert validate_subject("[fix]: " + "x" * 66) is not None
