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


def _fmt_table_value(v: object) -> str:
    """RLinf metric-table value formatting."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        a = abs(v)
        if a < 0.001 and v != 0:
            return f"{v:.2e}"
        if a < 0.01:
            return f"{v:.4f}"
        if a > 10000:
            return f"{v:.2e}"
        if a > 100:
            return f"{v:.1f}"
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


_BAR_WIDTH = 20


def _progress_bar(current: int, total: int, width: int = _BAR_WIDTH) -> str:
    """Solid/light block bar, e.g. ``████░░░░``; one line, no carriage return."""
    filled = max(0, min(width, int(round(width * current / total))))
    return "█" * filled + "░" * (width - filled)


def format_progress_line(
    desc: str,
    current: int,
    total: int | None,
    *,
    elapsed_s: float,
    eta_s: float | None,
    rate: float,
    unit: str = "it",
    status: str | None = None,
) -> str:
    """RLinf-style one-line progress string.

    total>0  -> "desc cur/total (pct%) · elapsed<eta · rate unit/s"
    total in (None, <=0) -> "desc cur · elapsed · rate unit/s" (open-ended).
    """
    if total and total > 0:
        pct = int(round(100.0 * current / total))
        head = f"{desc} [{_progress_bar(current, total)}] {current}/{total} ({pct}%)"
        timing = f"{_fmt_duration(elapsed_s)}<{_fmt_duration(eta_s or 0.0)}"
    else:
        head = f"{desc} {current}"
        timing = _fmt_duration(elapsed_s)
    line = f"{head} · {timing} · {rate:.1f} {unit}/s"
    if status:
        line = f"{line} · {status}"
    return line


def _fmt_table_duration(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _fit_table_line(text: str, width: int) -> str:
    if len(text) <= width:
        return text + (" " * (width - len(text)))
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _table_section_title(title: str, width: int) -> str:
    inner = width - 2
    title_text = f" {title} "
    if len(title_text) > inner:
        title_text = title_text[:inner]
    padding = inner - len(title_text)
    left = padding // 2
    right = padding - left
    return f"├{'─' * left}{title_text}{'─' * right}┤"


def _metric_table_categories(metrics: dict) -> dict[str, dict[str, object]]:
    categories: dict[str, dict[str, object]] = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }
    category_map = {
        "time": "Time",
        "env": "Environment",
        "rollout": "Rollout",
        "eval": "Evaluation",
        "replay_buffer": "Replay Buffer",
    }

    for key, value in metrics.items():
        if isinstance(value, str) or "/" not in str(key):
            continue
        category, metric_name = str(key).split("/", 1)
        if category in category_map:
            categories[category_map[category]][metric_name] = value
            continue
        if category != "train":
            continue
        if metric_name.startswith("actor/"):
            categories["Training/Actor"][metric_name] = value
        elif metric_name.startswith("critic/"):
            categories["Training/Critic"][metric_name] = value
        elif metric_name.startswith("replay_buffer/"):
            categories["Replay Buffer"][
                metric_name.replace("replay_buffer/", "", 1)
            ] = value
        else:
            categories["Training/Other"][metric_name] = value
    return categories


def format_metric_table(
    *,
    step: int,
    total_steps: int,
    elapsed_s: float,
    metrics: dict,
    start_step: int = 0,
    width: int = 120,
) -> str:
    """Return an RLinf-style global-step metric table.

    This mirrors RLinf's embodied progress output: a global-step progress
    header followed by grouped metric sections. It is pure so runners can test
    rendering without touching stdout.
    """
    total_steps = max(1, int(total_steps))
    step = int(step)
    start_step = int(start_step)
    width = max(40, int(width))
    progress = max(0.0, min(100.0, (step + 1) / total_steps * 100.0))
    steps_done = max(1, step + 1 - start_step)
    eta_s = elapsed_s / steps_done * max(0, total_steps - step - 1)

    bar_width = 40
    filled = int(bar_width * progress / 100.0)
    bar = "█" * filled + "░" * (bar_width - filled)
    inner = width - 2

    lines = [f"╭{'─' * inner}╮", _table_section_title("Metric Table", width)]
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line = f"│ {step_str} │ {progress_str}"
    lines.append(f"{_fit_table_line(line, inner)} │")

    elapsed_str = f"Elapsed: {_fmt_table_duration(elapsed_s)}"
    eta_str = f"ETA: {_fmt_table_duration(eta_s)}"
    step_time_str = f"Step Time: {elapsed_s / steps_done:.3f}s"
    line = f"│ {elapsed_str} │ {eta_str} │ {step_time_str}"
    lines.append(f"{_fit_table_line(line, inner)} │")

    categories = _metric_table_categories(metrics)
    table_width = width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if not category_metrics:
            continue
        lines.append(_table_section_title(category_name, width))
        lines.append(f"│{' ' * inner}│")
        sorted_metrics = sorted(category_metrics.items())
        for i in range(0, len(sorted_metrics), 3):
            row_metrics = []
            for j in range(3):
                if i + j >= len(sorted_metrics):
                    row_metrics.append("")
                    continue
                metric_name, metric_value = sorted_metrics[i + j]
                row_metrics.append(f"{metric_name}={_fmt_table_value(metric_value)}")
            lines.append(
                f"│{_fit_table_line(row_metrics[0], col_widths[0])}"
                f"│{_fit_table_line(row_metrics[1], col_widths[1])}"
                f"│{_fit_table_line(row_metrics[2], col_widths[2])}│"
            )
        lines.append(f"│{' ' * inner}│")

    lines.append(f"╰{'─' * inner}╯")
    return "\n".join(lines)
