"""Validate DreamerVLA's ``[type]: description`` commit-message contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_COMMIT_TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "init",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
_SUBJECT_PATTERN = re.compile(rf"^\[({'|'.join(ALLOWED_COMMIT_TYPES)})\]: [^\s].*$")
MAX_SUBJECT_LENGTH = 72


def validate_subject(subject: str) -> str | None:
    """Return an actionable error for an invalid subject, otherwise ``None``."""

    if not _SUBJECT_PATTERN.fullmatch(subject):
        allowed = ", ".join(ALLOWED_COMMIT_TYPES)
        return f"commit subject must match '[type]: description' (types: {allowed})"
    if len(subject) > MAX_SUBJECT_LENGTH:
        return f"commit subject must be at most {MAX_SUBJECT_LENGTH} characters"
    return None


def main(argv: list[str] | None = None) -> int:
    """Validate the subject in the commit-message file passed by pre-commit."""

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: check_commit_message.py <commit-message-file>", file=sys.stderr)
        return 2
    message_path = Path(arguments[0])
    subject = message_path.read_text(encoding="utf-8").splitlines()[0].strip()
    error = validate_subject(subject)
    if error is None:
        return 0
    print(f"{error}; got {subject!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
