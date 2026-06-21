"""Deterministic terminal-rendering helpers for training output.

Pure functions only — no I/O — so they are unit-testable. Value formatting
mirrors the threshold rules used by RLinf's print_metrics_table.
"""

from __future__ import annotations


def fmt_value(v: object) -> str:
    """Format a metric value compactly and deterministically."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        a = abs(v)
        if a < 0.001 or a >= 100000:
            return f"{v:.2e}"
        if a < 0.01:
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def phase_banner(
    title: str, *, subtitle: str | None = None, done: bool = False, width: int = 65
) -> str:
    """Return a single `===` banner line of exactly `width` chars."""
    label = title if not done else f"{title} — done"
    if subtitle:
        label = f"{label} · {subtitle}"
    label = f" {label} "
    if len(label) >= width - 2:
        label = label[: width - 2]
    pad = width - len(label)
    left = pad // 2
    right = pad - left
    return ("=" * left) + label + ("=" * right)


def metric_box(header: str, rows: list[str], *, width: int = 65) -> str:
    """Return a box-drawn metric panel; every line is exactly `width` chars."""
    inner = width - 2

    def _fit(text: str) -> str:
        if len(text) <= inner:
            return text + (" " * (inner - len(text)))
        return text[: inner - 1] + "…"

    head = f" {header} "
    if len(head) > inner:
        head = head[:inner]
    pad = inner - len(head)
    top = "╭" + ("─" * (pad // 2)) + head + ("─" * (pad - pad // 2)) + "╮"
    body = ["│" + _fit(r) + "│" for r in rows]
    bottom = "╰" + ("─" * inner) + "╯"
    return "\n".join([top, *body, bottom])


def count_trainable(module) -> int:
    """Number of trainable (requires_grad) parameters in a torch module.

    Returns 0 for a None module so callers can pass optional components.
    """
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _fmt_duration(seconds: float) -> str:
    """Format a duration as mm:ss, or h:mm:ss past an hour."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def format_progress_line(
    desc: str,
    current: int,
    total: int | None,
    *,
    elapsed_s: float,
    eta_s: float | None,
    rate: float,
    unit: str = "it",
) -> str:
    """RLinf-style one-line progress string.

    total>0  -> "desc cur/total (pct%) · elapsed<eta · rate unit/s"
    total in (None, <=0) -> "desc cur · elapsed · rate unit/s" (open-ended).
    """
    if total and total > 0:
        pct = int(round(100.0 * current / total))
        head = f"{desc} {current}/{total} ({pct}%)"
        timing = f"{_fmt_duration(elapsed_s)}<{_fmt_duration(eta_s or 0.0)}"
    else:
        head = f"{desc} {current}"
        timing = _fmt_duration(elapsed_s)
    return f"{head} · {timing} · {rate:.1f} {unit}/s"
