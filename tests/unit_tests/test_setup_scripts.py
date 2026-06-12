from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_shell_entrypoints_are_self_contained() -> None:
    root = _project_root()
    libero_entrypoints = (
        "scripts/train_vla.sh",
        "scripts/train_wm.sh",
        "scripts/train_dreamervla.sh",
        "scripts/eval_libero_vla.sh",
        "scripts/preprocess_libero.sh",
        "scripts/preprocess/prepare_libero_data.sh",
    )
    release_entrypoints = (
        *libero_entrypoints,
        "scripts/download_assets.sh",
        "scripts/install_env.sh",
        "scripts/preprocess/process_all_libero_data.sh",
    )
    for relpath in release_entrypoints:
        text = (root / relpath).read_text(encoding="utf-8")
        assert re.search(r"^\s*source\s+.*common_env\.sh", text, re.MULTILINE) is None, relpath
        assert "DVLA_ROOT" in text, relpath
        assert "DVLA_DATA_ROOT" in text, relpath

    for relpath in libero_entrypoints:
        text = (root / relpath).read_text(encoding="utf-8")
        assert 'LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"' in text, relpath
        assert "datasets: ${DVLA_DATA_ROOT}/datasets/libero" in text, relpath


def test_setup_and_download_scripts_are_release_entrypoints() -> None:
    root = _project_root()
    install = root / "scripts" / "install_env.sh"
    download = root / "scripts" / "download_assets.sh"
    install_steps = [
        root / "scripts" / "install" / name
        for name in (
            "00_apt_tools.sh",
            "10_conda_env.sh",
            "20_python_deps.sh",
            "30_third_party.sh",
            "40_verify.sh",
        )
    ]
    download_steps = [
        root / "scripts" / "download" / name
        for name in (
            "_env.sh",
            "10_worldvla.sh",
            "20_lumina.sh",
            "30_rynnvla.sh",
            "40_libero_dataset.sh",
            "50_calvin_dataset.sh",
        )
    ]

    assert install.is_file()
    assert download.is_file()
    assert all(step.is_file() for step in install_steps)
    assert all(step.is_file() for step in download_steps)

    install_text = install.read_text(encoding="utf-8")
    assert "INSTALL_STATE_DIR" in install_text
    assert "run_step" in install_text
    assert ".done" in install_text
    assert "00_apt_tools.sh" in install_text
    assert "10_conda_env.sh" in install_text
    assert "20_python_deps.sh" in install_text
    assert "30_third_party.sh" in install_text
    assert "40_verify.sh" in install_text
    assert "sudo apt" not in install_text
    assert "uv pip install" not in install_text

    step_text = "\n".join(step.read_text(encoding="utf-8") for step in install_steps)
    assert "apt update" in step_text
    assert "conda create -n" in step_text
    assert "uv pip install" in step_text
    assert "torch==2.5.1" in step_text
    assert "FLASH_ATTN" in step_text or "flash-attn" in step_text
    assert "third_party/LIBERO" in step_text
    assert "egl_probe" in step_text

    download_text = download.read_text(encoding="utf-8")
    assert "DOWNLOAD_STEPS" in download_text
    assert "10_worldvla.sh" in download_text
    assert "20_lumina.sh" in download_text
    assert "30_rynnvla.sh" in download_text
    assert "40_libero_dataset.sh" in download_text
    assert "50_calvin_dataset.sh" in download_text
    assert "hf download" not in download_text

    download_step_text = "\n".join(step.read_text(encoding="utf-8") for step in download_steps)
    assert "hf download" in download_step_text
    assert "Alibaba-DAMO-Academy/WorldVLA" in download_step_text
    assert "Alpha-VLLM/Lumina-mGPT-7B-768" in download_step_text
    assert "Alibaba-DAMO-Academy/RynnVLA-002" in download_step_text
    assert "download_libero_datasets.py" in download_step_text
    assert '--download-dir "${LIBERO_DATASET_DIR}"' in download_step_text
    assert 'CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"' in download_step_text
    assert "calvin" in download_step_text.lower()


def test_apt_install_step_handles_hosts_without_sudo() -> None:
    root = _project_root()
    apt_step = root / "scripts" / "install" / "00_apt_tools.sh"
    text = apt_step.read_text(encoding="utf-8")

    assert "command -v sudo" in text
    assert "APT_RUNNER" in text
    assert "INSTALL_APT_TOOLS=0" in text
    assert "sudo apt" not in text


def test_requirements_keep_runtime_dependency_set_curated() -> None:
    root = _project_root()
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    package_names = {
        re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0].replace("_", "-").lower()
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "gym" in package_names
    assert "easydict" in package_names
    assert "ray" not in package_names
    assert "tensorflow" not in package_names
    assert "tensorflow-datasets" not in package_names
    assert "torchdata" not in package_names
    assert "webdataset" not in package_names


def test_libero_data_script_defaults_to_his1_len_action1_and_filter_noops() -> None:
    root = _project_root()
    process_all = root / "scripts" / "preprocess" / "process_all_libero_data.sh"
    prepare = root / "scripts" / "preprocess" / "prepare_libero_data.sh"

    assert prepare.is_file()
    process_text = process_all.read_text(encoding="utf-8")
    prepare_text = prepare.read_text(encoding="utf-8")

    assert 'HIS="${HIS:-1}"' in process_text
    assert 'ACTION_HORIZON="${ACTION_HORIZON:-1}"' in process_text
    assert "GPUS=4,5" not in process_text
    assert 'TASK="${TASK:-libero_goal}"' in prepare_text
    assert 'FILTER_NOOPS="${FILTER_NOOPS:-1}"' in prepare_text
    assert 'RAW_LIBERO_DIR="${RAW_LIBERO_DIR:-${DVLA_DATA_ROOT}/datasets/libero/${TASK}}"' in prepare_text
    assert 'PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${DVLA_DATA_ROOT}/processed_data}"' in prepare_text
    assert "${TASK}_marked_t_${IMAGE_RESOLUTION}" in prepare_text
    assert "${TASK}_no_noops_t_${IMAGE_RESOLUTION}" in prepare_text


def test_top_level_preprocess_libero_wrapper_uses_repo_root_and_data_root() -> None:
    root = _project_root()
    wrapper = root / "scripts" / "preprocess_libero.sh"

    assert wrapper.is_file()
    text = wrapper.read_text(encoding="utf-8")

    assert 'export DVLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"' in text
    assert 'export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"' in text
    assert 'DEFAULT_SUITES=(libero_goal libero_object libero_spatial libero_10)' in text
    assert 'bash "${DVLA_ROOT}/scripts/preprocess/prepare_libero_data.sh"' in text
    assert 'TASK="${suite}"' in text


def test_release_scripts_do_not_ship_common_env() -> None:
    root = _project_root()

    assert not (root / "scripts" / "common_env.sh").exists()


def test_release_scripts_tree_is_curated() -> None:
    root = _project_root()
    scripts = root / "scripts"

    top_level_files = {path.name for path in scripts.iterdir() if path.is_file()}
    top_level_dirs = {path.name for path in scripts.iterdir() if path.is_dir()}

    assert top_level_files == {
        "README.md",
        "download_assets.sh",
        "eval_libero_vla.sh",
        "install_env.sh",
        "preprocess_libero.sh",
        "train_dreamervla.sh",
        "train_vla.sh",
        "train_wm.sh",
    }
    assert top_level_dirs == {
        "download",
        "eval",
        "install",
        "preprocess",
    }
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in gitignore
    assert "*.pyc" in gitignore


def test_scripts_tree_does_not_ship_python_modules() -> None:
    root = _project_root()
    scripts = root / "scripts"
    python_files = sorted(
        str(path.relative_to(root))
        for path in scripts.rglob("*.py")
        if "__pycache__" not in path.parts
    )

    assert python_files == []


def test_release_shell_scripts_launch_package_modules_with_python_m() -> None:
    root = _project_root()
    active_shells = sorted(
        path
        for path in (root / "scripts").rglob("*.sh")
        if "archive" not in path.relative_to(root / "scripts").parts
    )
    path_script_re = re.compile(r"dreamer_vla/[^\s\"']+\.py")
    offenders = {
        str(path.relative_to(root)): path_script_re.findall(path.read_text(encoding="utf-8"))
        for path in active_shells
        if path_script_re.search(path.read_text(encoding="utf-8"))
    }

    assert offenders == {}


def test_release_scripts_are_registered() -> None:
    root = _project_root()
    scripts = root / "scripts"
    registry = (scripts / "README.md").read_text(encoding="utf-8")
    registered = set(re.findall(r"`([^`]+)`", registry))
    allowed_unregistered = {"README.md"}
    script_files = sorted(
        path
        for path in scripts.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".sh", ".md"}
    )
    offenders = [
        str(path.relative_to(scripts))
        for path in script_files
        if str(path.relative_to(scripts)) not in registered
        and str(path.relative_to(scripts)) not in allowed_unregistered
    ]

    assert offenders == []


def test_portable_data_layout_manifest_exists_and_is_linked() -> None:
    root = _project_root()
    manifest = root / "docs" / "data_layout.md"
    setup = (root / "SETUP.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")

    assert manifest.is_file()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "${DVLA_DATA_ROOT}/datasets/libero/<suite>" in manifest_text
    assert "${DVLA_DATA_ROOT}/checkpoints" in manifest_text
    assert "${DVLA_DATA_ROOT}/processed_data" in manifest_text
    assert "scripts/download_assets.sh" in manifest_text
    assert "scripts/download/40_libero_dataset.sh" in manifest_text
    assert "scripts/preprocess/prepare_libero_data.sh" in manifest_text
    assert "DVLA_DATA_ROOT does not need to live inside DVLA_ROOT" in manifest_text
    assert "docs/data_layout.md" in setup
    assert "docs/data_layout.md" in readme
    assert "docs/data_layout.md" in scripts_readme


def test_release_docs_and_scripts_do_not_couple_data_root_to_repo_root() -> None:
    root = _project_root()
    active_paths = [
        root / "README.md",
        root / "README.zh-CN.md",
        root / "SETUP.md",
        root / "docs" / "install.md",
        root / "docs" / "data_layout.md",
        root / "scripts" / "README.md",
        *(
            path
            for path in (root / "scripts").rglob("*.sh")
            if "archive" not in path.relative_to(root / "scripts").parts
        ),
    ]
    coupled_defaults = [
        str(path.relative_to(root))
        for path in active_paths
        if '${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}' in path.read_text(encoding="utf-8")
    ]

    assert coupled_defaults == []


def test_active_shell_scripts_do_not_pin_machine_local_environment() -> None:
    root = _project_root()
    forbidden = (
        "/" + "/".join(("mnt", "data", "spoil", "workspace", "DreamerVLA")),
        "/" + "/".join(("home", "user01", "miniconda3", "envs", "dreamervla")),
    )
    active_scripts = sorted(
        path
        for path in (root / "scripts").rglob("*.sh")
        if "archive" not in path.relative_to(root / "scripts").parts
    )

    offenders: dict[str, list[str]] = {}
    for script in active_scripts:
        text = script.read_text(encoding="utf-8")
        matches = [needle for needle in forbidden if needle in text]
        if matches:
            offenders[str(script.relative_to(root))] = matches

    assert offenders == {}


def test_active_entrypoints_use_canonical_data_directory_names() -> None:
    root = _project_root()
    active_paths = [
        *root.glob("configs/*.yaml"),
        *root.glob("configs/task/*.yaml"),
        root / "README.md",
        root / "SETUP.md",
        root / "docs" / "data_layout.md",
        root / "scripts" / "README.md",
        *(
            path
            for path in (root / "scripts").rglob("*")
            if path.is_file()
            and path.suffix in {".sh", ".md"}
            and "archive" not in path.relative_to(root / "scripts").parts
            and "__pycache__" not in path.parts
        ),
    ]
    old_path_re = re.compile(r"(?:data|\$\{DVLA_DATA_ROOT\})/(?:ckpts|dataset)\b")
    offenders = {
        str(path.relative_to(root)): old_path_re.findall(path.read_text(encoding="utf-8"))
        for path in active_paths
        if old_path_re.search(path.read_text(encoding="utf-8"))
    }

    assert offenders == {}


def test_stable_docs_do_not_recommend_machine_specific_wrappers() -> None:
    root = _project_root()
    docs = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "SETUP.md": (root / "SETUP.md").read_text(encoding="utf-8"),
        "scripts/README.md": (root / "scripts" / "README.md").read_text(encoding="utf-8"),
    }
    machine_specific_re = re.compile(r"(?:_45|g67)\.sh")
    offenders = {
        relpath: machine_specific_re.findall(text)
        for relpath, text in docs.items()
        if machine_specific_re.search(text)
    }

    assert offenders == {}


def test_stable_docs_use_release_language() -> None:
    root = _project_root()
    docs = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "SETUP.md": (root / "SETUP.md").read_text(encoding="utf-8"),
        "scripts/README.md": (root / "scripts" / "README.md").read_text(encoding="utf-8"),
        "docs/data_layout.md": (root / "docs" / "data_layout.md").read_text(encoding="utf-8"),
    }
    banned_re = re.compile(
        r"formal|legacy-only|archive|archived|Historical|Machine-specific|正式|不再|历史计划|过程性",
        re.IGNORECASE,
    )
    offenders = {
        relpath: sorted(set(banned_re.findall(text)))
        for relpath, text in docs.items()
        if banned_re.search(text)
    }

    assert offenders == {}


def test_active_scripts_do_not_include_machine_specific_wrappers() -> None:
    root = _project_root()
    active_shells = [
        path
        for path in (root / "scripts").rglob("*.sh")
        if "archive" not in path.relative_to(root / "scripts").parts
    ]
    machine_specific_name_re = re.compile(r"(?:_45|g\d+)\.sh$")
    offenders = [str(path.relative_to(root)) for path in active_shells if machine_specific_name_re.search(path.name)]

    assert offenders == []
