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
    """Number of trainable (requires_grad) parameters in a torch module."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
