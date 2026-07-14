from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REMOVED_UNDERSCORE_WM_ROUTE = "dino" + "_wm"
_REMOVED_COMPACT_WM_ROUTE = "dino" + "wm"


def _assert_no_removed_wm_wording(text: str) -> None:
    lower = text.lower()
    assert _REMOVED_UNDERSCORE_WM_ROUTE not in lower
    assert _REMOVED_COMPACT_WM_ROUTE not in lower


def _tracked_source_paths(
    project_root: Path,
    active_roots: list[Path],
    *,
    skip_paths: set[Path],
    skip_parts: set[str],
    checked_suffixes: set[str],
) -> list[Path]:
    rel_roots = [str(root.relative_to(project_root)) for root in active_roots]
    result = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "--", *rel_roots],
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for rel_path in result.stdout.splitlines():
        path = project_root / rel_path
        if not path.is_file() or path in skip_paths:
            continue
        relative_parts = path.relative_to(project_root).parts
        if any(part in skip_parts for part in relative_parts):
            continue
        if path.suffix not in checked_suffixes:
            continue
        paths.append(path)
    return paths


def test_hygiene_source_scan_ignores_untracked_files(tmp_path) -> None:
    project_root = tmp_path / "repo"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True)
    tracked = docs_dir / "tracked.md"
    untracked = docs_dir / "feishu.md"
    tracked.write_text("active source\n", encoding="utf-8")
    untracked.write_text("local note mentioning wovr\n", encoding="utf-8")

    subprocess.run(
        ["git", "-C", str(project_root), "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "add", "docs/tracked.md"],
        check=True,
        capture_output=True,
        text=True,
    )

    paths = list(
        _tracked_source_paths(
            project_root,
            [docs_dir],
            skip_paths=set(),
            skip_parts=set(),
            checked_suffixes={".md"},
        )
    )

    assert tracked in paths
    assert untracked not in paths


def test_docs_and_smoke_script_do_not_point_at_removed_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[2]

    readme = (project_root / "README.md").read_text(encoding="utf-8")
    scripts_readme = (project_root / "scripts" / "README.md").read_text(encoding="utf-8")
    eval_script = (
        project_root / "scripts" / "experiments" / "cotrain" / "eval.sh"
    ).read_text(encoding="utf-8")

    assert "eval_wm.sh" not in readme
    assert "pretokenize_sft_wm_vla_smoke" not in scripts_readme
    assert "prepare_latent_data.sh" not in scripts_readme
    assert "dreamervla.launchers.train" in eval_script


def test_cotrain_train_and_eval_entrypoints_are_documented() -> None:
    project_root = Path(__file__).resolve().parents[2]
    agents = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    readme_zh = (project_root / "README.zh-CN.md").read_text(encoding="utf-8")
    config_registry = (project_root / "configs" / "README.md").read_text(
        encoding="utf-8"
    )
    scripts = (project_root / "scripts" / "README.md").read_text(encoding="utf-8")

    train = "scripts/experiments/cotrain/train.sh"
    evaluate = "scripts/experiments/cotrain/eval.sh"
    for text in (agents, readme, readme_zh, config_registry, scripts):
        assert train in text
        assert evaluate in text


def test_openvla_mainline_uses_only_hidden_token_public_names() -> None:
    project_root = Path(__file__).resolve().parents[2]
    explicit_paths = [
        project_root / "AGENTS.md",
        project_root / "README.md",
        project_root / "README.zh-CN.md",
        project_root / "configs" / "README.md",
        project_root / "docs" / "PARAMETERS.md",
        project_root / "docs" / "reference" / "model_datasets" / "openvla_oft_libero_goal.md",
        project_root / "docs" / "tutorials" / "experiments" / "OpenVLA_Onetraj_LIBERO.md",
        project_root / "docs" / "tutorials" / "experiments" / "EXPLAINED.md",
        project_root / "spec" / "06_routes.md",
        project_root / "dreamervla" / "config.py",
        project_root / "dreamervla" / "launchers" / "coldstart_warmup_cotrain.py",
        project_root / "dreamervla" / "runners" / "collect_rollouts_runner.py",
        project_root / "dreamervla" / "runners" / "collect_parallel_rollouts.py",
        project_root / "dreamervla" / "runners" / "rollout_hidden_extractor.py",
        project_root / "dreamervla" / "runners" / "embodied_eval_runner.py",
        project_root / "dreamervla" / "preprocess" / "preprocess_oft_hidden_token.py",
        project_root / "scripts" / "preprocess" / "10_oft_hidden_token.sh",
    ]
    globbed_paths = [
        *sorted((project_root / "configs" / "task").glob("openvla_onetraj*.yaml")),
        *sorted((project_root / "configs" / "dreamervla").glob("openvla_onetraj*.yaml")),
        *sorted((project_root / "configs" / "experiment").glob("*openvla_onetraj*.yaml")),
    ]
    legacy_source = "input_" + "token_embedding"
    legacy_namespace = "input_" + "tokens"
    legacy_dir = "input_" + "token_dir"
    forbidden = (
        legacy_source,
        f"task.openvla_oft.{legacy_namespace}",
        f"task.openvla_oft.{legacy_dir}",
        "wm_obs_dim: " + "229376",
        "_oft_" + legacy_source + "_vla_policy_h1",
        "preprocess_oft_" + legacy_namespace,
        "35_oft_" + legacy_namespace,
    )
    offenders: dict[str, list[str]] = {}
    for path in [*explicit_paths, *globbed_paths]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = [item for item in forbidden if item in text or item in path.name]
        if matches:
            offenders[str(path.relative_to(project_root))] = matches

    assert offenders == {}
    assert (
        project_root / "dreamervla" / "preprocess" / "preprocess_oft_hidden_token.py"
    ).is_file()
    assert (
        project_root
        / "configs"
        / "scripts"
        / "preprocess"
        / "preprocess_oft_hidden_token.yaml"
    ).is_file()
    assert (
        project_root / "scripts" / "preprocess" / "10_oft_hidden_token.sh"
    ).is_file()
    assert not (
        project_root
        / "dreamervla"
        / "preprocess"
        / f"preprocess_oft_{legacy_namespace}.py"
    ).exists()
    assert not (
        project_root
        / "configs"
        / "scripts"
        / "preprocess"
        / f"preprocess_oft_{legacy_namespace}.yaml"
    ).exists()
    assert not (
        project_root / "scripts" / "preprocess" / f"35_oft_{legacy_namespace}.sh"
    ).exists()


def test_legacy_projected_token_source_is_confined_to_migration_adapter() -> None:
    project_root = Path(__file__).resolve().parents[2]
    legacy_source = "input_" + "token_embedding"
    allowed = {
        Path("dreamervla/preprocess/sidecar_schema.py"),
        Path("tests/unit_tests/test_openvla_oft_hidden_token_shape.py"),
    }
    offenders: list[str] = []
    for root_name in ("dreamervla", "configs", "scripts", "tests", "docs", "spec"):
        for path in (project_root / root_name).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".sh", ".md", ".tex"}:
                continue
            relative = path.relative_to(project_root)
            if legacy_source in path.read_text(encoding="utf-8", errors="ignore") and relative not in allowed:
                offenders.append(str(relative))

    assert offenders == []


def test_active_sources_do_not_use_removed_rl_route_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_word = "wo" + "vr"
    active_roots = [
        project_root / "AGENTS.md",
        project_root / "README.md",
        project_root / "README.zh-CN.md",
        project_root / "SETUP.md",
        project_root / "configs",
        project_root / "docs",
        project_root / "dreamervla",
        project_root / "scripts",
        project_root / "spec",
    ]
    skip_paths = {
        project_root / "spec" / "99_manual_notes.md",
        project_root / "docs" / "rlinf_wovr_inference_optimizations.md",
    }
    skip_parts = {"__pycache__"}
    checked_suffixes = {".py", ".yaml", ".yml", ".md", ".sh", ".tex"}

    offenders: list[str] = []
    for path in _tracked_source_paths(
        project_root,
        active_roots,
        skip_paths=skip_paths,
        skip_parts=skip_parts,
        checked_suffixes=checked_suffixes,
    ):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if removed_word in text.lower() or removed_word in path.name.lower():
            offenders.append(str(path.relative_to(project_root)))

    assert offenders == []


def test_world_model_recipe_names_use_public_hyphenated_form() -> None:
    project_root = Path(__file__).resolve().parents[2]
    assert (project_root / "configs" / "worldmodel" / "dino-wm.yaml").is_file()
    assert (project_root / "configs" / "experiment" / "dino-wm.yaml").is_file()
    assert not (project_root / "configs" / "worldmodel" / "dino_wm.yaml").exists()
    assert not (project_root / "configs" / "experiment" / "dino_wm.yaml").exists()


def test_readme_documents_current_cotrain_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[2]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert "scripts/experiments/cotrain/train.sh" in readme
    assert "scripts/experiments/cotrain/eval.sh" in readme
    assert "train_wm.sh" not in readme
    assert "train_vla.sh" not in readme


def test_setup_guide_documents_current_cotrain_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[2]
    setup = (project_root / "SETUP.md").read_text(encoding="utf-8")

    assert "scripts/experiments/cotrain/train.sh" in setup
    assert "scripts/experiments/cotrain/eval.sh" in setup
    assert "train_wm.sh" not in setup
    assert "train_vla.sh" not in setup
    _assert_no_removed_wm_wording(setup)


def test_configs_readme_documents_current_cotrain_recipes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    configs_readme = (project_root / "configs" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "openvla_onetraj_libero_cotrain" in configs_readme
    assert "openvla_onetraj_libero_cotrain" in configs_readme
    assert "wm_full_dataset_train" in configs_readme
    assert "eval_libero_vla" in configs_readme
    _assert_no_removed_wm_wording(configs_readme)


def test_scripts_readme_documents_current_cotrain_launchers() -> None:
    project_root = Path(__file__).resolve().parents[2]
    scripts_readme = (project_root / "scripts" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "experiments/cotrain/train.sh" in scripts_readme
    assert "experiments/cotrain/eval.sh" in scripts_readme
    _assert_no_removed_wm_wording(scripts_readme)


def test_route_reference_documents_current_release_routes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    route_reference = (project_root / "docs" / "reference" / "routes.md").read_text(
        encoding="utf-8"
    )

    assert "collect_rollouts" in route_reference
    assert "dreamervla_wmcls_cotrain" in route_reference
    assert "eval_cotrain" in route_reference
    assert "wm_full_dataset_train" in route_reference
    _assert_no_removed_wm_wording(route_reference)


def test_experiment_tutorial_index_documents_current_recipes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tutorial_index = (
        project_root / "docs" / "tutorials" / "experiments" / "README.md"
    ).read_text(encoding="utf-8")

    assert "openvla_onetraj_libero_cotrain" in tutorial_index
    assert "wm_full_dataset_train" in tutorial_index
    assert "eval_libero_vla" in tutorial_index
    _assert_no_removed_wm_wording(tutorial_index)


def test_retired_model_tutorial_is_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_tutorial = ("Rynn" + "VLA") + "_LIBERO.md"
    assert not (project_root / "docs" / "tutorials" / "experiments" / removed_tutorial).exists()


def test_openvla_onetraj_tutorial_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tutorial = (
        project_root
        / "docs"
        / "tutorials"
        / "experiments"
        / "OpenVLA_Onetraj_LIBERO.md"
    ).read_text(encoding="utf-8")

    assert "scripts/experiments/cotrain/train.sh" in tutorial
    assert "scripts/experiments/cotrain/eval.sh" in tutorial
    assert "experiment=collect_rollouts" in tutorial
    _assert_no_removed_wm_wording(tutorial)


def test_parameter_reference_uses_role_based_wm_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parameter_reference = (project_root / "docs" / "PARAMETERS.md").read_text(
        encoding="utf-8"
    )

    assert "## World Model" in parameter_reference
    assert "world_model.chunk_rollout_chunks" in parameter_reference
    _assert_no_removed_wm_wording(parameter_reference)


def test_repository_structure_documents_current_release_routes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repository_structure = (
        project_root / "docs" / "repository_structure.md"
    ).read_text(encoding="utf-8")

    assert "collect_rollouts" in repository_structure
    assert "openvla_onetraj_libero_cotrain" in repository_structure
    assert "dreamervla_wmcls_cotrain" in repository_structure
    assert "eval_cotrain" in repository_structure
    _assert_no_removed_wm_wording(repository_structure)


def test_retired_model_dataset_reference_is_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_reference = ("rynn" + "vla") + "_libero_goal.md"
    assert not (
        project_root / "docs" / "reference" / "model_datasets" / removed_reference
    ).exists()


def test_openvla_model_dataset_reference_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reference = (
        project_root
        / "docs"
        / "reference"
        / "model_datasets"
        / "openvla_oft_libero_goal.md"
    ).read_text(encoding="utf-8")

    assert "scripts/experiments/cotrain/train.sh" in reference
    assert "scripts/experiments/cotrain/eval.sh" in reference
    _assert_no_removed_wm_wording(reference)


def test_experiment_explainer_uses_role_based_wm_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    explainer = (
        project_root / "docs" / "tutorials" / "experiments" / "EXPLAINED.md"
    ).read_text(encoding="utf-8")

    assert "WM chunk predictor" in explainer
    assert "WM paradigm" in explainer
    _assert_no_removed_wm_wording(explainer)


def test_removed_observation_diagnostics_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    diagnostics = project_root / "dreamervla" / "diagnostics"
    for name in (
        "compare_action_chunks.py",
        "diagnose_dreamervla_latent_distribution.py",
        "diagnose_ppo_imagine_vs_real.py",
        "diagnose_residual_cosine.py",
    ):
        assert not (diagnostics / name).exists(), name


def test_chunkwm_closeloop_diagnostic_usage_uses_role_based_wm_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root
        / "dreamervla"
        / "diagnostics"
        / "eval_chunkwm_closeloop.py"
    ).read_text(encoding="utf-8")

    assert "--ckpt /path/to/wm_run/ckpt/latest.ckpt" in source
    assert f"{_REMOVED_COMPACT_WM_ROUTE}_chunk" not in source


def test_active_docs_and_launchers_only_reference_existing_route_configs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"
    active_text_files = [
        project_root / "AGENTS.md",
        project_root / "CLAUDE.md",
        project_root / "README.md",
        config_dir / "README.md",
        project_root / "scripts" / "README.md",
        project_root / "dreamervla" / "train.py",
        project_root / "scripts" / "experiments" / "cotrain" / "train.sh",
        project_root / "scripts" / "experiments" / "cotrain" / "eval.sh",
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
            if not (config_dir / "scripts" / f"{route_name}.yaml").is_file()
            if not (config_dir / "experiment" / f"{route_name}.yaml").is_file()
        )
        assert missing == [], f"{text_file.relative_to(project_root)}: {missing}"

        removed_route_names = {
            "world_model_rssm_step",
            "dreamervla_" + "pi" + "0" + "_hidden_token_head_actor",
            "pretokenize_vla_libero_goal",
            "pretokenize_vla_libero_goal_" + "pi" + "0" + "_query",
            "rynn_backbone_dreamerv3_hidden_token_wm_libero_goal_precomputed",
        }
        stale = sorted(name for name in removed_route_names if name in text)
        assert stale == [], f"{text_file.relative_to(project_root)}: {stale}"


def test_agent_brief_describes_current_hydra_config_groups() -> None:
    project_root = Path(__file__).resolve().parents[2]
    agents_text = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    config_section = agents_text.split("- **`configs/`**", maxsplit=1)[1].split(
        "- **`scripts/`**", maxsplit=1
    )[0]

    for current_group in (
        "experiment/",
        "VLA/",
        "worldmodel/",
        "classifier/",
        "dreamervla/",
        "evaluation/",
        "task/",
        "logger/",
    ):
        assert current_group in config_section

    for stale_group in (
        "route/",
        "runner/",
        "dataset/",
        "world_model/",
        "algorithm/",
        "dataloader/",
    ):
        assert stale_group not in config_section


def test_claude_brief_delegates_to_current_agent_guidance() -> None:
    project_root = Path(__file__).resolve().parents[2]
    claude_text = (project_root / "CLAUDE.md").read_text(encoding="utf-8")

    assert "AGENTS.md" in claude_text
    assert "experiment=<name>" in claude_text
    assert "logger=tensorboard_wandb" in claude_text
    assert "logger=wandb" in claude_text
    assert "--config-name" not in claude_text
    assert "one top-level YAML per training route" not in claude_text


def test_active_sources_do_not_reference_removed_action_head_variant() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_variant = "pi0" + "_query"
    active_roots = [
        project_root / "AGENTS.md",
        project_root / "README.md",
        project_root / "configs",
        project_root / "docs",
        project_root / "dreamervla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skip_parts = {"__pycache__"}
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
    assert 'include = ["dreamervla*"]' in text


def test_files_live_under_their_architecture_domains() -> None:
    project_root = Path(__file__).resolve().parents[2]

    expected_top_level_dirs = {
        "configs",
        "docs",
        "dreamervla",
        "scripts",
        "tests",
    }
    for dirname in expected_top_level_dirs:
        assert (project_root / dirname).is_dir(), dirname

    forbidden_top_level_dirs = {
        "graveyard",
        "LIBERO",
        "runner",
        "src",
        "workspace",
        "dependencies",
    }
    for dirname in forbidden_top_level_dirs:
        assert not (project_root / dirname).exists(), dirname
    tracked_logs = subprocess.run(
        ["git", "ls-files", "logs"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert tracked_logs == []

    assert not (project_root / ".claude").exists()
    assert not (project_root / ".cursor").exists()
    assert not (project_root / "data" / "libero_goal_metainfo.json").exists()

    for dirname in ("cli", "trainer", "smoke", "evaluation", "training"):
        assert not (project_root / "dreamervla" / dirname).exists(), dirname

    assert (project_root / "dreamervla" / "train.py").is_file()

    preprocess_dir = project_root / "dreamervla" / "preprocess"
    assert not (preprocess_dir / "convertsation.py").exists()
    assert not (preprocess_dir / "concate_record.py").exists()
    assert not (preprocess_dir / "concate_action_world_model_data_libero.py").exists()
    assert not (preprocess_dir / "concate_record_libero.sh").exists()
    assert not (preprocess_dir / "conversation.py").exists()
    assert not (preprocess_dir / "concat_record.py").exists()
    assert not (preprocess_dir / "concat_action_world_model_data_libero.py").exists()
    assert not (preprocess_dir / "collect_online_rollouts_for_classifier.py").exists()
    for legacy_path in (
        "runners/online_dreamervla.py",
        "runners/frozen_wm_actor_critic.py",
        "runners/collect_online_rollouts_for_classifier.py",
        "runners/_online_dreamervla_dist.py",
        "runners/_online_dreamervla_checkpoint.py",
        "diagnostics/eval_frozen_wm_actor.py",
    ):
        assert not (project_root / "dreamervla" / legacy_path).exists(), legacy_path
    assert not (preprocess_dir / "xllmx").exists()

    assert not (project_root / "dreamervla" / "utils" / "libero_utils").exists()
    assert not (project_root / "dreamervla" / "models" / "xllmx").exists()
    assert not (project_root / "dreamervla" / "models" / "openvla-oft").exists()
    assert (preprocess_dir / "libero_utils").is_dir()
    assert (
        project_root / "dreamervla" / "models" / "embodiment" / "openvla_oft"
    ).is_dir()
    assert (
        project_root / "dreamervla" / "models" / "embodiment" / "chameleon_model"
    ).is_dir()

    assert not (project_root / "scripts" / "paper_tables").exists()
    assert not (project_root / "scripts" / "wm_variants_v4_v4E").exists()
    assert not (project_root / "scripts" / "process_all_libero_data.sh").exists()
    assert not (project_root / "scripts" / "eval_chunkwm_closeloop.py").exists()
    assert not (project_root / "scripts" / "eval" / "eval_libero_compat.py").exists()
    assert (project_root / "scripts" / "preprocess" / "process_all_libero_data.sh").is_file()
    assert not (project_root / "scripts" / "diagnostics").exists()
    diagnostics_dir = project_root / "dreamervla" / "diagnostics"
    assert (diagnostics_dir / "eval_chunkwm_closeloop.py").is_file()
    assert (diagnostics_dir / "eval_openvla_oft_libero.py").is_file()
    assert (diagnostics_dir / "openvla_oft_obs_action_policy.py").is_file()
    assert (diagnostics_dir / "smoke_libero_online_env.py").is_file()

    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    for ignored in (
        ".local/",
        ".claude/",
        ".cursor/",
        "data/",
        "third_party/",
        "docs/TODO.md",
        "docs/task_plan.md",
        "docs/*_plan.md",
    ):
        assert ignored in gitignore


def test_active_targets_use_canonical_module_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_files = sorted((project_root / "configs").glob("*.yaml"))

    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert "dreamervla.models.vla_actor" not in text, path.relative_to(project_root)
        assert "dreamervla.models.vla_policy" not in text, path.relative_to(project_root)


def test_preprocess_libero_utils_wrapper_is_removed() -> None:
    project_root = Path(__file__).resolve().parents[2]
    compat_path = project_root / "dreamervla" / "preprocess" / "libero_utils" / "libero_utils.py"
    assert not compat_path.exists()


def test_removed_56x1024_preprocess_stack_is_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed = (
        "dreamervla/models/embodiment/rynnvla_runtime.py",
        "dreamervla/models/embodiment/rynnvla_image_ops.py",
        "dreamervla/preprocess/item_processor.py",
        "dreamervla/preprocess/pre_tokenize_action_local.py",
        "dreamervla/preprocess/pre_tokenize_action_state_local.py",
        "dreamervla/preprocess/pretoken_state_action_model.py",
        "dreamervla/preprocess/xllmx",
    )
    for relative in removed:
        assert not (project_root / relative).exists(), relative


def test_production_experiments_do_not_embed_test_only_workers() -> None:
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / "configs" / "experiment"

    assert not (experiment_dir / "cotrain_tiny.yaml").exists()
    offenders = [
        path.relative_to(project_root)
        for path in experiment_dir.glob("*.yaml")
        if "._test_" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_online_replay_is_library_module_not_cli_local_class() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner_path = project_root / "dreamervla" / "runners" / "world_model_training_runner.py"
    runner_text = runner_path.read_text(encoding="utf-8")

    assert (project_root / "dreamervla" / "runtime" / "online_replay.py").is_file()
    assert "from dreamervla.runtime.online_replay import" in runner_text
    assert "class OnlineReplay" not in runner_text
    assert "def pack_replay_task_stats_for_ddp" not in runner_text
    assert "def unpack_replay_task_stats_from_ddp" not in runner_text


def test_distributed_training_helper_lives_with_runtime() -> None:
    project_root = Path(__file__).resolve().parents[2]
    helper_path = project_root / "dreamervla" / "runtime" / "distributed.py"

    assert helper_path.is_file()
    assert "class NopretokenizeSFTDistributedHelper" in helper_path.read_text(encoding="utf-8")
    assert not (project_root / "dreamervla" / "trainer").exists()

    runner_import_offenders: dict[str, str] = {}
    old_trainer_import = "dreamervla." + "trainer"
    for path in (project_root / "dreamervla" / "runners").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if old_trainer_import in text:
            runner_import_offenders[str(path.relative_to(project_root))] = old_trainer_import
    assert runner_import_offenders == {}


def test_package_has_no_redundant_top_level_command_groups() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_roots = [
        project_root / "AGENTS.md",
        project_root / "CLAUDE.md",
        project_root / "README.md",
        project_root / "SETUP.md",
        project_root / "docs",
        project_root / "dreamervla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skipped_parts = {"__pycache__", "superpowers"}
    checked_suffixes = {".py", ".md", ".sh", ".yaml", ".yml"}
    old_paths = tuple(
        "dreamervla." + suffix
        for suffix in (
            "cli",
            "trainer",
            "training",
            "evaluation",
            "smoke",
        )
    )
    offenders: dict[str, list[str]] = {}
    for root in active_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in checked_suffixes:
                continue
            if any(part in skipped_parts for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            matches = [old_path for old_path in old_paths if old_path in text]
            if matches:
                offenders[str(path.relative_to(project_root))] = matches

    assert offenders == {}


def test_chameleon_lazy_exports_only_existing_modeling_symbols() -> None:
    project_root = Path(__file__).resolve().parents[2]
    init_path = (
        project_root
        / "dreamervla"
        / "models"
        / "embodiment"
        / "chameleon_model"
        / "chameleon"
        / "__init__.py"
    )
    text = init_path.read_text(encoding="utf-8")

    assert "ChameleonForConditionalGeneration_ContinuousHead" not in text


def test_models_package_does_not_hide_import_failures() -> None:
    project_root = Path(__file__).resolve().parents[2]
    text = (project_root / "dreamervla" / "models" / "__init__.py").read_text(encoding="utf-8")

    assert "except Exception" not in text
    assert "= None" not in text


def test_package_modules_do_not_insert_project_root_into_sys_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (project_root / "dreamervla").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "sys.path.insert(0, str(PROJECT_ROOT))" in text:
            offenders.append(str(path.relative_to(project_root)))

    assert offenders == []


def test_active_configs_do_not_describe_ignored_targets() -> None:
    project_root = Path(__file__).resolve().parents[2]
    offenders: dict[str, list[str]] = {}
    banned = (
        "online script ignores",
        "NOT this config's _target_",
        "not part of the main Runner launch path",
    )
    for path in (project_root / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        matches = [item for item in banned if item in text]
        if matches:
            offenders[str(path.relative_to(project_root))] = matches

    assert offenders == {}


def test_active_configs_do_not_pin_machine_local_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"
    active_configs = sorted(
        path for path in config_dir.rglob("*.yaml") if path.is_file()
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
        project_root / "dreamervla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skip_parts = {"__pycache__"}
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


def test_source_package_data_helpers_are_not_gitignored() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_files = [
        "dreamervla/preprocess/preprocess_oft_hidden_token.py",
        "dreamervla/preprocess/sidecar_schema.py",
    ]

    missing = [path for path in source_files if not (project_root / path).is_file()]
    assert missing == []

    result = subprocess.run(
        ["git", "check-ignore", *source_files],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr


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
        "dreamervla.workspace",
        "dreamervla.dataloader",
        "dreamervla.env.",
        "dreamervla.env import",
        "dreamervla/workspace",
        "dreamervla/dataloader",
        "dreamervla/env/",
    ]
    for path in active_docs:
        text = path.read_text(encoding="utf-8")
        stale = [pattern for pattern in stale_patterns if pattern in text]
        assert stale == [], f"{path.relative_to(project_root)}: {stale}"
