from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_spec_superpowers_plans_are_marked_historical_reference() -> None:
    plan_dir = ROOT / "spec" / "superpowers" / "plans"
    plans = sorted(plan_dir.glob("*.md"))
    assert plans

    readme = ROOT / "spec" / "superpowers" / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "historical/reference" in lowered
    assert "not current architecture source of truth" in lowered

    spec_index = (ROOT / "spec" / "README.md").read_text(encoding="utf-8")
    assert "superpowers/README.md" in spec_index


def test_spec_prompt_file_is_marked_reference_only() -> None:
    prompt = ROOT / "spec" / "98_prompt.md"
    text = prompt.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "historical/reference prompt context" in lowered
    assert "not current architecture source of truth" in lowered

    spec_index = (ROOT / "spec" / "README.md").read_text(encoding="utf-8")
    assert "98_prompt.md" in spec_index
    assert "reference-only" in spec_index.lower()


def test_replay_buffer_metric_namespace_is_documented() -> None:
    spec_text = (ROOT / "spec" / "03_coding_style.md").read_text(
        encoding="utf-8"
    )
    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "`replay_buffer/`" in spec_text
    assert "`replay_buffer/`" in agents_text
