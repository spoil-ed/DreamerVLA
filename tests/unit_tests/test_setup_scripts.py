from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_formal_shell_entrypoints_source_common_env() -> None:
    root = _project_root()
    expected = 'source "${SCRIPT_DIR}/common_env.sh"'
    for relpath in (
        "scripts/train_vla.sh",
        "scripts/train_wm.sh",
        "scripts/train_dreamervla.sh",
        "scripts/eval_libero_vla.sh",
    ):
        text = (root / relpath).read_text(encoding="utf-8")
        assert expected in text, relpath


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
    assert "${TASK}_marked_t_${IMAGE_RESOLUTION}" in prepare_text
    assert "${TASK}_no_noops_t_${IMAGE_RESOLUTION}" in prepare_text


def test_active_shell_scripts_do_not_pin_machine_local_environment() -> None:
    root = _project_root()
    forbidden = (
        "/mnt/data/spoil/workspace/DreamerVLA",
        "/home/user01/miniconda3/envs/dreamervla",
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
