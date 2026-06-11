from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_formal_shell_entrypoints_are_self_contained() -> None:
    root = _project_root()
    libero_entrypoints = (
        "scripts/train_vla.sh",
        "scripts/train_wm.sh",
        "scripts/train_dreamervla.sh",
        "scripts/eval_libero_vla.sh",
        "scripts/preprocess/prepare_libero_data.sh",
    )
    formal_entrypoints = (
        *libero_entrypoints,
        "scripts/download_assets.sh",
        "scripts/install_env.sh",
        "scripts/preprocess/process_all_libero_data.sh",
    )
    for relpath in formal_entrypoints:
        text = (root / relpath).read_text(encoding="utf-8")
        assert re.search(r"^\s*source\s+.*common_env\.sh", text, re.MULTILINE) is None, relpath
        assert "DVLA_ROOT" in text, relpath
        assert "DVLA_DATA_ROOT" in text, relpath

    for relpath in libero_entrypoints:
        text = (root / relpath).read_text(encoding="utf-8")
        assert 'LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"' in text, relpath
        assert "datasets: ${DVLA_DATA_ROOT}/dataset/libero" in text, relpath


def test_setup_and_download_scripts_are_formal_entrypoints() -> None:
    root = _project_root()
    install = root / "scripts" / "install_env.sh"
    download = root / "scripts" / "download_assets.sh"

    assert install.is_file()
    assert download.is_file()

    install_text = install.read_text(encoding="utf-8")
    assert "sudo apt" in install_text
    assert "conda create -n" in install_text
    assert "uv pip install" in install_text
    assert "torch==2.5.1" in install_text
    assert "flash_attn" in install_text
    assert "third_party/LIBERO" in install_text
    assert "egl_probe" in install_text

    download_text = download.read_text(encoding="utf-8")
    assert "hf download" in download_text
    assert "download_libero_datasets.py" in download_text
    assert '--download-dir "${LIBERO_DATASET_DIR}"' in download_text
    assert 'CKPT_DIR="${DVLA_DATA_ROOT}/ckpts"' in download_text
    assert "calvin" in download_text.lower()


def test_libero_data_script_defaults_to_his1_len_action1_and_filter_noops() -> None:
    root = _project_root()
    process_all = root / "scripts" / "preprocess" / "process_all_libero_data.sh"
    prepare = root / "scripts" / "preprocess" / "prepare_libero_data.sh"

    assert prepare.is_file()
    process_text = process_all.read_text(encoding="utf-8")
    prepare_text = prepare.read_text(encoding="utf-8")

    assert 'HIS="${HIS:-1}"' in process_text
    assert 'ACTION_HORIZON="${ACTION_HORIZON:-1}"' in process_text
    assert 'TASK="${TASK:-libero_goal}"' in prepare_text
    assert 'FILTER_NOOPS="${FILTER_NOOPS:-1}"' in prepare_text
    assert 'RAW_LIBERO_DIR="${RAW_LIBERO_DIR:-${DVLA_DATA_ROOT}/dataset/libero/${TASK}}"' in prepare_text
    assert 'PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${DVLA_DATA_ROOT}/processed_data}"' in prepare_text
    assert "${TASK}_marked_t_${IMAGE_RESOLUTION}" in prepare_text
    assert "${TASK}_no_noops_t_${IMAGE_RESOLUTION}" in prepare_text


def test_common_env_is_marked_legacy_only() -> None:
    root = _project_root()
    text = (root / "scripts" / "common_env.sh").read_text(encoding="utf-8")

    assert "DEPRECATED for formal entrypoints" in text
    assert "DVLA_DATA_ROOT" in text


def test_portable_data_layout_manifest_exists_and_is_linked() -> None:
    root = _project_root()
    manifest = root / "docs" / "data_layout.md"
    setup = (root / "SETUP.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")

    assert manifest.is_file()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "${DVLA_DATA_ROOT}/dataset/libero/<suite>" in manifest_text
    assert "${DVLA_DATA_ROOT}/processed_data" in manifest_text
    assert "scripts/download_assets.sh" in manifest_text
    assert "scripts/preprocess/prepare_libero_data.sh" in manifest_text
    assert "docs/data_layout.md" in setup
    assert "docs/data_layout.md" in readme
    assert "docs/data_layout.md" in scripts_readme


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
