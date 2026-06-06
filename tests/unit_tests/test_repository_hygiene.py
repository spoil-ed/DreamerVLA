from __future__ import annotations

from pathlib import Path
import re


def test_docs_and_smoke_script_do_not_point_at_removed_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[2]

    readme = (project_root / "README.md").read_text(encoding="utf-8")
    scripts_readme = (project_root / "scripts" / "README.md").read_text(
        encoding="utf-8"
    )
    train_script = (project_root / "scripts" / "train_vla.sh").read_text(
        encoding="utf-8"
    )
    eval_script = (project_root / "scripts" / "eval_libero_vla.sh").read_text(
        encoding="utf-8"
    )

    assert "eval_wm.sh" not in readme
    assert "pretokenize_sft_wm_vla_smoke" not in scripts_readme
    assert "prepare_latent_data.sh" not in scripts_readme
    assert "-m dreamer_vla.cli.train" in train_script
    assert "CONFIG=\"${CONFIG:-eval_libero_vla}\"" in eval_script
    assert "-m dreamer_vla.cli.train" in eval_script
    assert (
        project_root
        / "third_party"
        / "LIBERO"
        / "benchmark_scripts"
        / "download_libero_datasets.py"
    ).is_file()


def test_active_docs_and_launchers_only_reference_existing_route_configs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"
    active_text_files = [
        project_root / "AGENTS.md",
        project_root / "CLAUDE.md",
        project_root / "README.md",
        config_dir / "README.md",
        project_root / "scripts" / "README.md",
        project_root / "dreamer_vla" / "cli" / "train.py",
        project_root / "scripts" / "train_wm.sh",
        project_root / "scripts" / "train_dreamervla.sh",
        project_root / "scripts" / "training" / "train_online_rynnvla_action_hidden_dreamervla.py",
        project_root / "scripts" / "training" / "train_frozen_wm_actor_critic.py",
        project_root / "scripts" / "eval" / "eval_frozen_wm_actor.py",
        project_root / "scripts" / "diagnostics" / "compare_action_chunks.py",
    ]

    for text_file in active_text_files:
        text = text_file.read_text(encoding="utf-8")
        route_names = set(re.findall(r"\bconfigs/([A-Za-z0-9_]+)\.yaml\b", text))
        route_names.update(re.findall(r"--config-name[ =]([A-Za-z0-9_]+)", text))
        route_names.update(re.findall(r"\bCONFIG(?:_NAME)?=([A-Za-z0-9_]+)\b", text))
        route_names.update(re.findall(r"\$\{CONFIG:-([A-Za-z0-9_]+)\}", text))
        missing = sorted(
            route_name
            for route_name in route_names
            if route_name != "CONFIG"
            if not (config_dir / f"{route_name}.yaml").is_file()
        )
        assert missing == [], f"{text_file.relative_to(project_root)}: {missing}"

        removed_route_names = {
            "world_model_rssm_step",
            "dreamervla_pi0_action_hidden_head_actor",
            "pretokenize_vla_libero_goal",
            "pretokenize_vla_libero_goal_" + "pi0" + "_query",
            "rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed",
        }
        stale = sorted(name for name in removed_route_names if name in text)
        assert stale == [], f"{text_file.relative_to(project_root)}: {stale}"


def test_active_sources_do_not_reference_removed_action_head_variant() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_variant = "pi0" + "_query"
    active_roots = [
        project_root / "AGENTS.md",
        project_root / "README.md",
        project_root / "configs",
        project_root / "docs",
        project_root / "dreamer_vla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skip_parts = {"archive", "__pycache__"}
    checked_suffixes = {".py", ".yaml", ".yml", ".md", ".sh", ".tex"}

    offenders: list[str] = []
    for root in active_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            if any(part in skip_parts for part in path.parts):
                continue
            if path.suffix not in checked_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if removed_variant in text or removed_variant in path.name:
                offenders.append(str(path.relative_to(project_root)))

    assert offenders == []


def test_repository_structure_doc_and_editable_package_metadata_exist() -> None:
    project_root = Path(__file__).resolve().parents[2]

    assert (project_root / "docs" / "repository_structure.md").is_file()

    pyproject = project_root / "pyproject.toml"
    assert pyproject.is_file()
    text = pyproject.read_text(encoding="utf-8")
    assert 'name = "dreamer-vla"' in text
    assert 'include = ["dreamer_vla*"]' in text


def test_files_live_under_their_architecture_domains() -> None:
    project_root = Path(__file__).resolve().parents[2]

    expected_top_level_dirs = {
        "configs",
        "data",
        "docs",
        "dreamer_vla",
        "scripts",
        "tests",
        "third_party",
    }
    for dirname in expected_top_level_dirs:
        assert (project_root / dirname).is_dir(), dirname

    forbidden_top_level_dirs = {
        "graveyard",
        "LIBERO",
        "logs",
        "runner",
        "src",
        "workspace",
        "dependencies",
    }
    for dirname in forbidden_top_level_dirs:
        assert not (project_root / dirname).exists(), dirname

    assert (project_root / "docs" / "archive" / "graveyard").is_dir()
    assert (project_root / "data" / "outputs" / "logs" / "root_legacy_logs").is_dir()

    cli_files = {
        path.name for path in (project_root / "dreamer_vla" / "cli").glob("*.py")
    }
    assert cli_files == {"__init__.py", "train.py"}

    preprocess_dir = project_root / "dreamer_vla" / "preprocess"
    assert not (preprocess_dir / "convertsation.py").exists()
    assert not (preprocess_dir / "concate_record.py").exists()
    assert not (preprocess_dir / "concate_action_world_model_data_libero.py").exists()
    assert not (preprocess_dir / "concate_record_libero.sh").exists()
    assert (preprocess_dir / "conversation.py").is_file()
    assert (preprocess_dir / "concat_record.py").is_file()
    assert (preprocess_dir / "concat_action_world_model_data_libero.py").is_file()
    assert (preprocess_dir / "xllmx").is_dir()

    assert not (project_root / "dreamer_vla" / "utils" / "libero_utils").exists()
    assert not (project_root / "dreamer_vla" / "models" / "xllmx").exists()
    assert not (project_root / "dreamer_vla" / "models" / "openvla-oft").exists()
    assert (preprocess_dir / "libero_utils").is_dir()
    assert (project_root / "third_party" / "openvla-oft-lightweight").is_dir()

    assert not (project_root / "scripts" / "process_all_libero_data.sh").exists()
    assert not (project_root / "scripts" / "eval_chunkwm_closeloop.py").exists()
    assert (project_root / "scripts" / "preprocess" / "process_all_libero_data.sh").is_file()
    assert (project_root / "scripts" / "diagnostics" / "eval_chunkwm_closeloop.py").is_file()

    assert (project_root / "scripts" / "eval" / "eval_libero_legacy.py").is_file()


def test_active_targets_use_canonical_module_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_files = [
        *sorted((project_root / "configs").glob("*.yaml")),
        project_root / "scripts" / "diagnostics" / "diagnose_dreamervla_latent_distribution.py",
    ]

    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert "dreamer_vla.models.vla_actor" not in text, path.relative_to(project_root)
        assert "dreamer_vla.models.vla_policy" not in text, path.relative_to(project_root)


def test_active_configs_do_not_pin_machine_local_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"
    active_configs = sorted(
        path
        for path in config_dir.rglob("*.yaml")
        if "archive" not in path.relative_to(config_dir).parts
    )

    forbidden_roots = [
        "/" + "/".join(("mnt", "data", "spoil", "workspace", "DreamerVLA")),
        "/" + "/".join(("home", "user01")),
    ]
    for path in active_configs:
        text = path.read_text(encoding="utf-8")
        stale = [root for root in forbidden_roots if root in text]
        assert stale == [], f"{path.relative_to(project_root)}: {stale}"


def test_active_files_do_not_pin_machine_local_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_roots = [
        project_root / "AGENTS.md",
        project_root / "CLAUDE.md",
        project_root / "CONTRIBUTING.md",
        project_root / "README.md",
        project_root / "README.zh-CN.md",
        project_root / "SETUP.md",
        project_root / "configs",
        project_root / "docs",
        project_root / "dreamer_vla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skip_parts = {"archive", "__pycache__"}
    checked_suffixes = {".py", ".yaml", ".yml", ".md", ".sh", ".tex"}
    forbidden_roots = [
        "/" + "mnt" + "/",
        "/" + "home" + "/",
    ]

    offenders: dict[str, list[str]] = {}
    for root in active_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            if any(part in skip_parts for part in path.relative_to(project_root).parts):
                continue
            if path.suffix not in checked_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            stale = [root for root in forbidden_roots if root in text]
            if stale:
                offenders[str(path.relative_to(project_root))] = stale

    assert offenders == {}


def test_active_docs_do_not_describe_removed_source_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_docs = [
        project_root / "AGENTS.md",
        project_root / "CLAUDE.md",
        project_root / "CONTRIBUTING.md",
        project_root / "README.md",
        project_root / "docs" / "repository_structure.md",
        project_root / "scripts" / "README.md",
    ]

    stale_patterns = [
        "from src.",
        "import src.",
        "dreamer_vla.workspace",
        "dreamer_vla.dataloader",
        "dreamer_vla.env.",
        "dreamer_vla.env import",
        "dreamer_vla/workspace",
        "dreamer_vla/dataloader",
        "dreamer_vla/env/",
    ]
    for path in active_docs:
        text = path.read_text(encoding="utf-8")
        stale = [pattern for pattern in stale_patterns if pattern in text]
        assert stale == [], f"{path.relative_to(project_root)}: {stale}"
