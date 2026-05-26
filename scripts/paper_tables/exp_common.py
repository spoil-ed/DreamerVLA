#!/usr/bin/env python3
"""Shared helpers for DreamerVLA paper experiment interfaces.

These helpers intentionally stay lightweight: the expensive work is still done
by the existing training/eval scripts.  The table-specific interfaces use this
module to build launch commands, collect JSON metrics, and export LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "paper_tables"


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def read_json(path: str | Path) -> Any:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    out = project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_clean_json(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return out


def nested_get(data: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if as_float(v) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def success_rate_from_json(data: Mapping[str, Any]) -> float | None:
    for key in ("eval_success_rate", "success_rate", "real_success", "long_success", "success"):
        value = as_float(data.get(key))
        if value is not None:
            return value
    successes = as_float(data.get("eval_total_successes", data.get("total_successes")))
    episodes = as_float(data.get("eval_total_episodes", data.get("total_episodes")))
    if successes is not None and episodes and episodes > 0:
        return successes / episodes
    return None


def collect_record_values(records: Sequence[Mapping[str, Any]], mode: str, key: str) -> float | None:
    vals = []
    for record in records:
        mode_payload = record.get(mode)
        if isinstance(mode_payload, Mapping):
            value = as_float(mode_payload.get(key))
            if value is not None:
                vals.append(value)
    return mean(vals)


def latex_escape(text: Any) -> str:
    raw = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in raw)


def format_metric(value: Any, *, pct: bool = False, precision: int = 1) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "--"
    if pct:
        return f"{100.0 * numeric:.{precision}f}"
    return f"{numeric:.{precision}f}"


def write_latex_table(
    path: str | Path,
    *,
    caption: str,
    label: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    align: str | None = None,
) -> Path:
    out = project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    alignment = align or ("l" + "c" * (len(columns) - 1))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def command_to_str(command: Sequence[str], env: Mapping[str, str] | None = None) -> str:
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items()) + " "
    return env_prefix + " ".join(shlex.quote(str(part)) for part in command)


def run_or_print(command: Sequence[str], *, env: Mapping[str, str] | None, execute: bool) -> int:
    rendered = command_to_str(command, env)
    print(rendered, flush=True)
    if not execute:
        return 0
    merged_env = os.environ.copy()
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})
    return subprocess.call([str(part) for part in command], cwd=str(PROJECT_ROOT), env=merged_env)


class KeyValueAction(argparse.Action):
    """Parse repeated KEY=VALUE arguments into a dictionary."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        items = dict(getattr(namespace, self.dest, None) or {})
        for raw in values:
            if "=" not in raw:
                parser.error(f"{option_string} expects KEY=VALUE, got {raw!r}")
            key, value = raw.split("=", 1)
            items[key] = value
        setattr(namespace, self.dest, items)


def add_common_collect_args(parser: argparse.ArgumentParser, default_json: Path, default_tex: Path) -> None:
    parser.add_argument("--out-json", default=str(default_json))
    parser.add_argument("--out-tex", default=str(default_tex))


def parse_metric_triples(items: Sequence[Sequence[str]] | None) -> list[tuple[str, str, Path]]:
    triples: list[tuple[str, str, Path]] = []
    for item in items or []:
        name, key, path = item
        triples.append((name, key, project_path(path)))
    return triples

