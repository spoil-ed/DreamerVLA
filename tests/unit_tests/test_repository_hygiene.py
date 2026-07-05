from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REMOVED_UNDERSCORE_WM_ROUTE = "dino" + "_wm"
_REMOVED_COMPACT_WM_ROUTE = "dino" + "wm"
_REMOVED_DASHED_WM_LABEL = "DINO" + "-WM"


def _assert_no_removed_wm_wording(text: str) -> None:
    lower = text.lower()
    assert _REMOVED_UNDERSCORE_WM_ROUTE not in lower
    assert _REMOVED_COMPACT_WM_ROUTE not in lower
    assert _REMOVED_DASHED_WM_LABEL not in text


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
    eval_script = (project_root / "scripts" / "eval_libero_vla.sh").read_text(encoding="utf-8")

    assert "eval_wm.sh" not in readme
    assert "pretokenize_sft_wm_vla_smoke" not in scripts_readme
    assert "prepare_latent_data.sh" not in scripts_readme
    assert "dreamervla.launchers.train" in eval_script


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
        # Background prompt context that documents the external reference
        # implementation it was derived from.
        project_root / "spec" / "98_prompt.md",
        # User-requested cross-project alignment analysis; the report must name
        # the external scheme it analyzes.
        project_root / "docs" / "rlinf_wovr_inference_optimizations.md",
    }
    skip_parts = {"archive", "__pycache__", "superpowers"}
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


def test_active_sources_do_not_use_removed_world_model_naming() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_fragments = (
        "dino" + "_wm",
        "dino" + "wm",
        "dino" + "-wm",
        "dino" + "wm" + "world" + "model",
        "chunkaware" + "dino" + "wm" + "world" + "model",
        "wm" + "world" + "model",
    )
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
        project_root / "tests" / "unit_tests",
    ]
    skip_paths = {
        project_root / "spec" / "99_manual_notes.md",
    }
    skip_parts = {"archive", "__pycache__"}
    checked_suffixes = {".py", ".yaml", ".yml", ".md", ".sh", ".tex"}

    offenders: list[str] = []
    for path in _tracked_source_paths(
        project_root,
        active_roots,
        skip_paths=skip_paths,
        skip_parts=skip_parts,
        checked_suffixes=checked_suffixes,
    ):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        path_name = path.name.lower()
        if any(fragment in text or fragment in path_name for fragment in removed_fragments):
            offenders.append(str(path.relative_to(project_root)))

    assert offenders == []


def test_readme_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert "experiment=world_model_chunk" in readme
    assert "experiment=dreamervla_rynn_wm_lumos" in readme
    assert f"experiment=world_model_{_REMOVED_COMPACT_WM_ROUTE}_chunk" not in readme
    assert (
        f"experiment=dreamervla_rynn_{_REMOVED_UNDERSCORE_WM_ROUTE}_lumos"
        not in readme
    )


def test_setup_guide_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    setup = (project_root / "SETUP.md").read_text(encoding="utf-8")

    assert "experiment=world_model_chunk" in setup
    assert "experiment=world_model_chunk_input_tokens" in setup
    assert "experiment=oft_world_model_chunk_input_tokens" in setup
    assert "experiment=oft_discrete_token_world_model_chunk" in setup
    assert "experiment=dreamervla_rynn_wm_lumos" in setup
    assert "experiment=dreamervla_rynn_wm_lumos_input_tokens" in setup
    _assert_no_removed_wm_wording(setup)


def test_configs_readme_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    configs_readme = (project_root / "configs" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "experiment=world_model_chunk" in configs_readme
    assert "dreamervla_rynn_wm_lumos" in configs_readme
    assert "oft_world_model_chunk" in configs_readme
    assert "dreamervla_oft_wm_lumos" in configs_readme
    _assert_no_removed_wm_wording(configs_readme)


def test_scripts_readme_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    scripts_readme = (project_root / "scripts" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "experiment=world_model_chunk" in scripts_readme
    _assert_no_removed_wm_wording(scripts_readme)


def test_route_reference_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    route_reference = (project_root / "docs" / "reference" / "routes.md").read_text(
        encoding="utf-8"
    )

    assert "world_model_chunk" in route_reference
    assert "dreamervla_rynn_wm_lumos" in route_reference
    assert "dreamervla_oft_discrete_token_wm_lumos" in route_reference
    _assert_no_removed_wm_wording(route_reference)


def test_experiment_tutorial_index_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tutorial_index = (
        project_root / "docs" / "tutorials" / "experiments" / "README.md"
    ).read_text(encoding="utf-8")

    assert "world_model_chunk" in tutorial_index
    assert "dreamervla_rynn_wm_lumos" in tutorial_index
    assert "oft_discrete_token_world_model_chunk" in tutorial_index
    assert "dreamervla_oft_discrete_token_wm_lumos" in tutorial_index
    _assert_no_removed_wm_wording(tutorial_index)


def test_rynnvla_tutorial_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tutorial = (
        project_root / "docs" / "tutorials" / "experiments" / "RynnVLA_LIBERO.md"
    ).read_text(encoding="utf-8")

    assert "experiment=world_model_chunk" in tutorial
    assert "experiment=dreamervla_rynn_wm_lumos" in tutorial
    assert "experiment=dreamervla_rynn_wm_actor_critic" in tutorial
    _assert_no_removed_wm_wording(tutorial)


def test_openvla_onetraj_tutorial_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tutorial = (
        project_root
        / "docs"
        / "tutorials"
        / "experiments"
        / "OpenVLA_Onetraj_LIBERO.md"
    ).read_text(encoding="utf-8")

    assert "e2e_coldstart_warmup_cotrain_ray.sh" in tutorial
    assert "experiment=collect_rollouts_ray" in tutorial
    _assert_no_removed_wm_wording(tutorial)


def test_parameter_reference_uses_role_based_wm_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parameter_reference = (project_root / "docs" / "PARAMETERS.md").read_text(
        encoding="utf-8"
    )

    assert "WM architecture" in parameter_reference
    assert "WM chunk predictor" in parameter_reference
    _assert_no_removed_wm_wording(parameter_reference)


def test_repository_structure_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repository_structure = (
        project_root / "docs" / "repository_structure.md"
    ).read_text(encoding="utf-8")

    assert "world_model_chunk" in repository_structure
    assert "dreamervla_rynn_wm_lumos" in repository_structure
    assert "dreamervla_oft_wm_lumos" in repository_structure
    _assert_no_removed_wm_wording(repository_structure)


def test_rynnvla_model_dataset_reference_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reference = (
        project_root / "docs" / "reference" / "model_datasets" / "rynnvla_libero_goal.md"
    ).read_text(encoding="utf-8")

    assert "WM token axis" in reference
    assert "experiment=world_model_chunk" in reference
    assert "dreamervla_rynn_wm_lumos" in reference
    _assert_no_removed_wm_wording(reference)


def test_openvla_model_dataset_reference_prefers_role_based_wm_route_examples() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reference = (
        project_root
        / "docs"
        / "reference"
        / "model_datasets"
        / "openvla_oft_libero_goal.md"
    ).read_text(encoding="utf-8")

    assert "experiment=oft_world_model_chunk" in reference
    assert "dreamervla_oft_wm_lumos" in reference
    _assert_no_removed_wm_wording(reference)


def test_experiment_explainer_uses_role_based_wm_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    explainer = (
        project_root / "docs" / "tutorials" / "experiments" / "EXPLAINED.md"
    ).read_text(encoding="utf-8")

    assert "WM chunk predictor" in explainer
    assert "WM paradigm" in explainer
    _assert_no_removed_wm_wording(explainer)


def test_ppo_imagine_diagnostic_docstring_uses_role_based_wm_wording() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root
        / "dreamervla"
        / "diagnostics"
        / "diagnose_ppo_imagine_vs_real.py"
    ).read_text(encoding="utf-8")

    assert "WM imagined PPO routes" in source
    _assert_no_removed_wm_wording(source)


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
        project_root / "scripts" / "train_dreamervla.sh",
        project_root / "dreamervla" / "runners" / "online_dreamervla.py",
        project_root / "dreamervla" / "runners" / "frozen_wm_actor_critic.py",
        project_root / "dreamervla" / "diagnostics" / "eval_frozen_wm_actor.py",
        project_root / "dreamervla" / "diagnostics" / "compare_action_chunks.py",
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
            "dreamervla_" + "pi" + "0" + "_action_hidden_head_actor",
            "pretokenize_vla_libero_goal",
            "pretokenize_vla_libero_goal_" + "pi" + "0" + "_query",
            "rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed",
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
    skip_parts = {"archive", "__pycache__", "superpowers"}
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
    assert not (project_root / "configs" / "archive").exists()
    docs_archive = project_root / "docs" / "archive"
    assert (docs_archive / "plans").is_dir()
    assert not [p for p in docs_archive.iterdir() if p.is_file()]
    assert not (project_root / "data" / "libero_goal_metainfo.json").exists()

    for dirname in ("cli", "trainer", "smoke", "evaluation", "training"):
        assert not (project_root / "dreamervla" / dirname).exists(), dirname

    assert (project_root / "dreamervla" / "train.py").is_file()

    preprocess_dir = project_root / "dreamervla" / "preprocess"
    assert not (preprocess_dir / "convertsation.py").exists()
    assert not (preprocess_dir / "concate_record.py").exists()
    assert not (preprocess_dir / "concate_action_world_model_data_libero.py").exists()
    assert not (preprocess_dir / "concate_record_libero.sh").exists()
    assert (preprocess_dir / "conversation.py").is_file()
    assert (preprocess_dir / "concat_record.py").is_file()
    assert (preprocess_dir / "concat_action_world_model_data_libero.py").is_file()
    assert not (preprocess_dir / "collect_online_rollouts_for_classifier.py").exists()
    assert (
        project_root / "dreamervla" / "runners" / "collect_online_rollouts_for_classifier.py"
    ).is_file()
    assert (preprocess_dir / "xllmx").is_dir()

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

    assert not (project_root / "scripts" / "archive").exists()
    assert not (project_root / "scripts" / "paper_tables").exists()
    assert not (project_root / "scripts" / "wm_variants_v4_v4E").exists()
    assert not (project_root / "scripts" / "process_all_libero_data.sh").exists()
    assert not (project_root / "scripts" / "eval_chunkwm_closeloop.py").exists()
    assert not (project_root / "scripts" / "eval" / "eval_libero_legacy.py").exists()
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
        "docs/superpowers/*",
        "docs/TODO.md",
        "docs/task_plan.md",
        "docs/*_plan.md",
    ):
        assert ignored in gitignore


def test_active_targets_use_canonical_module_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    active_files = [
        *sorted((project_root / "configs").glob("*.yaml")),
        project_root / "dreamervla" / "diagnostics" / "diagnose_dreamervla_latent_distribution.py",
    ]

    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert "dreamervla.models.vla_actor" not in text, path.relative_to(project_root)
        assert "dreamervla.models.vla_policy" not in text, path.relative_to(project_root)


def test_preprocess_libero_utils_reexports_canonical_env_helpers() -> None:
    project_root = Path(__file__).resolve().parents[2]
    compat_path = project_root / "dreamervla" / "preprocess" / "libero_utils" / "libero_utils.py"
    text = compat_path.read_text(encoding="utf-8")

    assert "from dreamervla.envs.libero_env import" in text
    assert "OffScreenRenderEnv" not in text
    assert "def get_libero_env" not in text
    assert "def get_libero_image" not in text
    assert "def quat2axisangle" not in text


def test_rynnvla_processor_shared_helpers_have_single_home() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runtime_path = project_root / "dreamervla" / "models" / "encoder" / "rynnvla_runtime.py"
    preprocess_path = project_root / "dreamervla" / "preprocess" / "item_processor.py"
    conversation_path = project_root / "dreamervla" / "preprocess" / "conversation.py"

    runtime_text = runtime_path.read_text(encoding="utf-8")
    preprocess_text = preprocess_path.read_text(encoding="utf-8")
    conversation_text = conversation_path.read_text(encoding="utf-8")

    assert "from dreamervla.utils.conversation import Conversation" in runtime_text
    assert "from dreamervla.utils.conversation import Conversation" in conversation_text
    for text, path in (
        (runtime_text, runtime_path),
        (preprocess_text, preprocess_path),
    ):
        assert "from dreamervla.models.encoder.rynnvla_image_ops import" in text, path.relative_to(
            project_root
        )
        assert "def center_crop" not in text, path.relative_to(project_root)
        assert "def var_center_crop" not in text, path.relative_to(project_root)
        assert "def generate_crop_size_list" not in text, path.relative_to(project_root)


def test_online_replay_is_library_module_not_cli_local_class() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cli_path = project_root / "dreamervla" / "runners" / "online_dreamervla.py"
    cli_text = cli_path.read_text(encoding="utf-8")

    assert (project_root / "dreamervla" / "runners" / "online_replay.py").is_file()
    assert "from dreamervla.runners.online_replay import" in cli_text
    assert "class OnlineReplay" not in cli_text
    assert "def pack_replay_task_stats_for_ddp" not in cli_text
    assert "def unpack_replay_task_stats_from_ddp" not in cli_text


def test_distributed_training_helper_lives_with_runners() -> None:
    project_root = Path(__file__).resolve().parents[2]
    helper_path = project_root / "dreamervla" / "runners" / "distributed.py"

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


def test_residual_cosine_diagnostic_has_no_import_time_io() -> None:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "dreamervla" / "diagnostics" / "diagnose_residual_cosine.py"
    text = path.read_text(encoding="utf-8")

    assert "def main(" in text
    assert 'if __name__ == "__main__":' in text
    before_main = text.split("def main(", 1)[0]
    assert "np.load(" not in before_main
    assert ".mkdir(" not in before_main
    assert "h5py.File(" not in before_main


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
        project_root / "dreamervla",
        project_root / "scripts",
        project_root / "tests",
    ]
    skip_parts = {"archive", "__pycache__", "superpowers"}
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
        "dreamervla/preprocess/xllmx/data/__init__.py",
        "dreamervla/preprocess/xllmx/data/data_reader.py",
        "dreamervla/preprocess/xllmx/data/item_processor.py",
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
