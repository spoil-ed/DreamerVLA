from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_spec_index_points_only_to_release_architecture_docs() -> None:
    spec_index = (ROOT / "spec" / "README.md").read_text(encoding="utf-8")

    assert "superpowers" not in spec_index
    assert "98_prompt" not in spec_index


def test_replay_buffer_metric_namespace_is_documented() -> None:
    spec_text = (ROOT / "spec" / "03_coding_style.md").read_text(encoding="utf-8")
    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "`replay_buffer/`" in spec_text
    assert "`replay_buffer/`" in agents_text


def test_manual_cotrain_resume_contract_is_replay_free() -> None:
    manual_notes = (ROOT / "spec" / "99_manual_notes.md").read_text(encoding="utf-8")

    assert "Replay 是临时运行态" in manual_notes
    assert "不得写入 cotrain checkpoint" in manual_notes
    assert "replay/sampling state 等实际存在的状态" not in manual_notes


def test_docs_index_local_links_exist() -> None:
    docs_dir = ROOT / "docs"
    index_text = (docs_dir / "README.md").read_text(encoding="utf-8")

    for target in re.findall(r"\]\(([^)]+)\)", index_text):
        if "://" in target or target.startswith("#"):
            continue
        assert (docs_dir / target).exists(), target


def test_e2e_readme_does_not_claim_populated_directory_is_empty() -> None:
    e2e_dir = ROOT / "tests" / "e2e_tests"
    readme = (e2e_dir / "README.md").read_text(encoding="utf-8")

    assert list(e2e_dir.glob("test_*.py"))
    assert "currently empty" not in readme


def test_active_route_docs_match_hydra_runner_targets() -> None:
    route_spec = (ROOT / "spec" / "06_routes.md").read_text(encoding="utf-8")
    route_reference = (ROOT / "docs" / "reference" / "routes.md").read_text(encoding="utf-8")
    tutorial = (
        ROOT / "docs" / "tutorials" / "experiments" / "OpenVLA_Onetraj_LIBERO.md"
    ).read_text(encoding="utf-8")

    assert "`openvla_libero` | `DreamerRunner`" in route_spec
    assert "`openvla_onetraj_libero_cotrain` | `CotrainRunner`" in route_spec
    assert "`experiment=openvla_libero` | `DreamerRunner`" in route_reference
    assert "`openvla_onetraj_libero_cotrain` | `CotrainRunner`" in route_reference
    assert "`dreamervla.runners.DreamerRunner`" in tutorial


def test_active_docs_do_not_reference_removed_frozen_model_launcher() -> None:
    active_docs = (
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "spec" / "06_routes.md",
    )

    for path in active_docs:
        assert "dreamervla.launchers.frozen_model_pre_mainline" not in path.read_text(
            encoding="utf-8"
        ), path.relative_to(ROOT)
