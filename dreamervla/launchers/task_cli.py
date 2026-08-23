"""Shared ``--task`` normalization for task-aware public launchers."""

from __future__ import annotations

from collections.abc import Collection, Sequence

LIBERO_TASKS = frozenset(
    {
        "libero_goal",
        "libero_object",
        "libero_spatial",
        "libero_10",
    }
)


def _override_key(value: str) -> str | None:
    if value.startswith("--") or "=" not in value:
        return None
    return value.split("=", 1)[0].lstrip("+~")


def normalize_task_flag(
    argv: Sequence[str],
    *,
    hydra_key: str,
    as_list: bool = False,
    valid_tasks: Collection[str] = LIBERO_TASKS,
) -> tuple[list[str], str | None]:
    """Remove one ``--task`` flag and return its equivalent Hydra override.

    Native ``key=value`` syntax remains supported. Combining both spellings is rejected
    so argument order cannot silently select a different suite.
    """

    remaining: list[str] = []
    task: str | None = None
    index = 0
    values = list(argv)
    while index < len(values):
        item = values[index]
        if item == "--task":
            if index + 1 >= len(values):
                raise SystemExit("--task requires a LIBERO suite")
            candidate = values[index + 1]
            if candidate.startswith("--"):
                raise SystemExit("--task requires a LIBERO suite")
            index += 2
        elif item.startswith("--task="):
            candidate = item.split("=", 1)[1]
            index += 1
        else:
            remaining.append(item)
            index += 1
            continue

        if task is not None:
            raise SystemExit("--task may be specified only once")
        if candidate not in valid_tasks:
            valid = ", ".join(sorted(valid_tasks))
            raise SystemExit(f"unknown LIBERO task {candidate!r}; valid: {valid}")
        task = candidate

    if task is None:
        return remaining, None
    if any(_override_key(item) == hydra_key for item in remaining):
        raise SystemExit(f"--task cannot be combined with the Hydra {hydra_key}=... override")
    rendered = f"[{task}]" if as_list else task
    return remaining, f"{hydra_key}={rendered}"


__all__ = ["LIBERO_TASKS", "normalize_task_flag"]
