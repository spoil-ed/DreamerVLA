from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_spec_index_points_only_to_release_architecture_docs() -> None:
    spec_index = (ROOT / "spec" / "README.md").read_text(encoding="utf-8")

    assert "superpowers" not in spec_index
    assert "98_prompt" not in spec_index


def test_replay_buffer_metric_namespace_is_documented() -> None:
    spec_text = (ROOT / "spec" / "03_coding_style.md").read_text(
        encoding="utf-8"
    )
    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "`replay_buffer/`" in spec_text
    assert "`replay_buffer/`" in agents_text
