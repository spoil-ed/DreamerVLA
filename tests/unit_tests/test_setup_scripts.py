from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_script_python_module_entrypoints_are_importable() -> None:
    root = _project_root()
    script_paths = sorted((root / "scripts" / "preprocess").glob("*.sh"))
    script_paths.extend(sorted((root / "scripts" / "experiments").rglob("*.sh")))

    missing: list[str] = []
    for path in script_paths:
        text = path.read_text(encoding="utf-8")
        modules = sorted(
            set(
                re.findall(
                    r"python\s+-m\s+(dreamervla(?:\.[A-Za-z_][A-Za-z0-9_]*)+)",
                    text,
                )
            )
        )
        for module in modules:
            if importlib.util.find_spec(module) is None:
                missing.append(f"{path.relative_to(root)}: {module}")

    assert missing == []


def _write_hdf5_reward_repair_python_stub(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${PYTHON_STUB_LOG}"\n'
        "module=''\n"
        "prev=''\n"
        'for arg in "$@"; do\n'
        '  if [[ "${prev}" == \'-m\' ]]; then module="${arg}"; fi\n'
        '  prev="${arg}"\n'
        "done\n"
        "get_value() {\n"
        '  local key="$1"\n'
        "  shift\n"
        "  local arg\n"
        '  for arg in "$@"; do\n'
        '    if [[ "${arg}" == "${key}="* ]]; then\n'
        "      printf '%s\\n' \"${arg#*=}\"\n"
        "      return 0\n"
        "    fi\n"
        "  done\n"
        "  return 1\n"
        "}\n"
        'case "${module}" in\n'
        "  dreamervla.preprocess.check_artifacts)\n"
        '    dir="$(get_value dir "$@" || true)"\n'
        '    if [[ -n "${EXPECTED_HDF5_DIR:-}" && "${dir}" == "${EXPECTED_HDF5_DIR}" && ! -f "${dir}/stub_demo.hdf5" ]]; then\n'
        "      exit 1\n"
        "    fi\n"
        '    if [[ -n "${EXPECTED_REWARD_DIR:-}" && "${dir}" == "${EXPECTED_REWARD_DIR}" && ! -f "${dir}/stub_demo.hdf5" ]]; then\n'
        "      exit 1\n"
        "    fi\n"
        "    ;;\n"
        "  dreamervla.preprocess.filter_marked_libero_hdf5)\n"
        '    out="$(get_value output_dir "$@")"\n'
        '    mkdir -p "${out}"\n'
        '    touch "${out}/stub_demo.hdf5"\n'
        "    ;;\n"
        "  dreamervla.preprocess.preprocess_remaining_steps_reward)\n"
        '    out="$(get_value output_dir "$@")"\n'
        '    mkdir -p "${out}"\n'
        '    touch "${out}/stub_demo.hdf5"\n'
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_install_verify_exports_dvla_root_to_python_diagnostics(tmp_path: Path) -> None:
    root = _project_root()
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda_stub = bin_dir / "conda"
    python_stub = bin_dir / "python"
    conda_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == 'shell.bash' && \"${2:-}\" == 'hook' ]]; then\n"
        "  printf '%s\\n' 'conda() { return 0; }'\n"
        "fi\n",
        encoding="utf-8",
    )
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'DVLA_ROOT=%s args=%s\\n\' "${DVLA_ROOT:-}" "$*" >> "${PYTHON_STUB_LOG}"\n'
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'dreamervla.diagnostics.checks.verify_install' ]]; then\n"
        '  test -n "${DVLA_ROOT:-}"\n'
        "fi\n",
        encoding="utf-8",
    )
    conda_stub.chmod(0o755)
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/install/60_verify.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"DVLA_ROOT={root}" in log_path.read_text(encoding="utf-8")


def test_lerobot_download_is_pinned_and_clears_proxy_environment(tmp_path: Path) -> None:
    root = _project_root()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "hf.log"
    hf_stub = bin_dir / "hf"
    hf_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'http_proxy=%s\\n\' "${http_proxy-unset}" >> "${HF_STUB_LOG}"\n'
        'printf \'https_proxy=%s\\n\' "${https_proxy-unset}" >> "${HF_STUB_LOG}"\n'
        'printf \'HTTP_PROXY=%s\\n\' "${HTTP_PROXY-unset}" >> "${HF_STUB_LOG}"\n'
        'printf \'HTTPS_PROXY=%s\\n\' "${HTTPS_PROXY-unset}" >> "${HF_STUB_LOG}"\n'
        'printf \'ALL_PROXY=%s\\n\' "${ALL_PROXY-unset}" >> "${HF_STUB_LOG}"\n'
        'printf \'args=%s\\n\' "$*" >> "${HF_STUB_LOG}"\n',
        encoding="utf-8",
    )
    hf_stub.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "HF_STUB_LOG": str(log_path),
            "DVLA_DATA_ROOT": str(tmp_path / "data"),
            "http_proxy": "http://vpn.invalid:1",
            "https_proxy": "http://vpn.invalid:2",
            "HTTP_PROXY": "http://vpn.invalid:3",
            "HTTPS_PROXY": "http://vpn.invalid:4",
            "ALL_PROXY": "socks5://vpn.invalid:5",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/download/20_libero_dataset.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert "http_proxy=unset" in log
    assert "https_proxy=unset" in log
    assert "HTTP_PROXY=unset" in log
    assert "HTTPS_PROXY=unset" in log
    assert "ALL_PROXY=unset" in log
    assert "download physical-intelligence/libero --repo-type dataset" in log
    assert "--revision a4336d589d589045d1c56423ffdf3b88a0e19b1f" in log
    assert str(tmp_path / "data/datasets/lerobot/physical-intelligence/libero") in log


def test_preprocess_launchers_accept_common_cli_flags(tmp_path: Path) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    reward_dir = data_root / "processed_data" / "libero_goal" / "no_noops_t_256_remaining_reward"
    reward_dir.mkdir(parents=True)
    (reward_dir / "demo.hdf5").touch()
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'dreamervla.launchers.workflow' ]]; then\n"
        '  exec "${REAL_PYTHON}" "$@"\n'
        "fi\n"
        'printf \'%s\\n\' "$*" >> "${PYTHON_STUB_LOG}"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_STUB_LOG": str(log_path),
            "REAL_PYTHON": sys.executable,
            "DVLA_DATA_ROOT": str(data_root),
            "OFT_FAKE_COMPONENTS": "1",
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "scripts/preprocess/prepare_libero_data.sh",
            "task=libero_goal",
            f"data_root={data_root}",
            "gpus=4,5",
            "num_procs=3",
            "overwrite=true",
            "only=[10_oft_hidden_token]",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    workflow_output = result.stdout + result.stderr
    log_text = log_path.read_text(encoding="utf-8")
    assert "[workflow:preprocess_suite]" in workflow_output
    assert "config=preprocess/preprocess_suite" in workflow_output
    assert "run 10_oft_hidden_token" in workflow_output

    dreamervla_calls = [
        line for line in log_text.splitlines() if line.startswith("-m dreamervla.preprocess.")
    ]
    assert dreamervla_calls
    assert "dreamervla.preprocess.preprocess_oft_hidden_token" in log_text
    assert not any(
        re.search(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", line) for line in dreamervla_calls
    )


def test_preprocess_steps_share_root_and_alias_artifact_mapping(tmp_path: Path) -> None:
    root = _project_root()
    preprocess_dir = root / "scripts" / "preprocess"
    for name in ("00_hdf5_reward.sh", "10_oft_hidden_token.sh", "20_validate.sh"):
        source = (preprocess_dir / name).read_text(encoding="utf-8")
        assert 'export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"' in source

    data_root = tmp_path / "data"
    artifact_name = "openvla_onetraj_libero_libero_goal"
    reward_dir = data_root / "processed_data" / artifact_name / "no_noops_t_256_remaining_reward"
    reward_dir.mkdir(parents=True)
    (reward_dir / "demo.hdf5").touch()

    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'%s\\n\' "$*" >> "${PYTHON_STUB_LOG}"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_ROOT": str(root),
            "DVLA_DATA_ROOT": str(data_root),
            "TASK": "libero_goal",
            "LIBERO_SUITE": "libero_goal",
            "TASK_NAME": "openvla_onetraj_libero",
            "OFT_FAKE_COMPONENTS": "1",
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/preprocess/10_oft_hidden_token.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(reward_dir) in log_path.read_text(encoding="utf-8")


def test_prepare_libero_data_rebuilds_empty_marked_dir(tmp_path: Path) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    raw_dir = data_root / "datasets" / "libero" / "libero_goal"
    processed = data_root / "processed_data" / "libero_goal"
    marked_dir = processed / "marked_t_256"
    hdf5_dir = processed / "no_noops_t_256"
    reward_dir = processed / "no_noops_t_256_remaining_reward"
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"

    raw_dir.mkdir(parents=True)
    (raw_dir / "placeholder_demo.hdf5").touch()
    marked_dir.mkdir(parents=True)
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${PYTHON_STUB_LOG}"\n'
        "module=''\n"
        "prev=''\n"
        'for arg in "$@"; do\n'
        '  if [[ "${prev}" == \'-m\' ]]; then module="${arg}"; fi\n'
        '  prev="${arg}"\n'
        "done\n"
        'case "${module}" in\n'
        "  dreamervla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op)\n"
        '    for arg in "$@"; do\n'
        '      if [[ "${arg}" == libero_target_dir=* ]]; then out="${arg#libero_target_dir=}"; mkdir -p "${out}"; touch "${out}/stub_demo.hdf5"; fi\n'
        "    done\n"
        "    ;;\n"
        "  dreamervla.preprocess.filter_marked_libero_hdf5)\n"
        '    for arg in "$@"; do\n'
        '      if [[ "${arg}" == output_dir=* ]]; then out="${arg#output_dir=}"; mkdir -p "${out}"; touch "${out}/stub_demo.hdf5"; fi\n'
        "    done\n"
        "    ;;\n"
        "  dreamervla.preprocess.preprocess_remaining_steps_reward)\n"
        '    for arg in "$@"; do\n'
        '      if [[ "${arg}" == output_dir=* ]]; then out="${arg#output_dir=}"; mkdir -p "${out}"; touch "${out}/stub_demo.hdf5"; fi\n'
        "    done\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_DATA_ROOT": str(data_root),
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "TASK": "libero_goal",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/preprocess/00_hdf5_reward.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any("regenerate_libero_dataset_filter_no_op" in call for call in calls)
    assert hdf5_dir.joinpath("stub_demo.hdf5").is_file()
    assert reward_dir.joinpath("stub_demo.hdf5").is_file()


def test_hdf5_reward_repairs_incomplete_filtered_stage_without_full_overwrite(
    tmp_path: Path,
) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    raw_dir = data_root / "datasets" / "libero" / "libero_goal"
    processed = data_root / "processed_data" / "libero_goal"
    marked_dir = processed / "marked_t_256"
    hdf5_dir = processed / "no_noops_t_256"
    reward_dir = processed / "no_noops_t_256_remaining_reward"
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"

    raw_dir.mkdir(parents=True)
    (raw_dir / "placeholder_demo.hdf5").touch()
    marked_dir.mkdir(parents=True)
    (marked_dir / "stub_demo.hdf5").touch()
    hdf5_dir.mkdir(parents=True)
    stale_hdf5 = hdf5_dir / "r0_shard_000.hdf5"
    stale_hdf5.touch()
    _write_hdf5_reward_repair_python_stub(python_stub)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_DATA_ROOT": str(data_root),
            "EXPECTED_HDF5_DIR": str(hdf5_dir),
            "EXPECTED_REWARD_DIR": str(reward_dir),
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "TASK": "libero_goal",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/preprocess/00_hdf5_reward.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[00_hdf5_reward] repair incomplete filtered stage" in result.stderr
    assert hdf5_dir.joinpath("stub_demo.hdf5").is_file()
    assert not stale_hdf5.exists()
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any("filter_marked_libero_hdf5" in call for call in calls)
    assert not any("regenerate_libero_dataset_filter_no_op" in call for call in calls)


def test_hdf5_reward_repairs_incomplete_reward_stage_without_full_overwrite(
    tmp_path: Path,
) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    raw_dir = data_root / "datasets" / "libero" / "libero_goal"
    processed = data_root / "processed_data" / "libero_goal"
    marked_dir = processed / "marked_t_256"
    hdf5_dir = processed / "no_noops_t_256"
    reward_dir = processed / "no_noops_t_256_remaining_reward"
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"

    raw_dir.mkdir(parents=True)
    (raw_dir / "placeholder_demo.hdf5").touch()
    marked_dir.mkdir(parents=True)
    (marked_dir / "stub_demo.hdf5").touch()
    hdf5_dir.mkdir(parents=True)
    (hdf5_dir / "stub_demo.hdf5").touch()
    reward_dir.mkdir(parents=True)
    stale_reward = reward_dir / "r0_shard_000.hdf5"
    stale_reward.touch()
    _write_hdf5_reward_repair_python_stub(python_stub)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_DATA_ROOT": str(data_root),
            "EXPECTED_HDF5_DIR": str(hdf5_dir),
            "EXPECTED_REWARD_DIR": str(reward_dir),
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "TASK": "libero_goal",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/preprocess/00_hdf5_reward.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[00_hdf5_reward] repair incomplete reward stage" in result.stderr
    assert reward_dir.joinpath("stub_demo.hdf5").is_file()
    assert not stale_reward.exists()
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any("preprocess_remaining_steps_reward" in call for call in calls)
    assert not any("filter_marked_libero_hdf5" in call for call in calls)
    assert not any("regenerate_libero_dataset_filter_no_op" in call for call in calls)


def test_prepare_libero_data_rejects_empty_raw_dir_before_generation(tmp_path: Path) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    raw_dir = data_root / "datasets" / "libero" / "libero_goal"
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"

    raw_dir.mkdir(parents=True)
    python_stub.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'%s\\n\' "$*" >> "${PYTHON_STUB_LOG}"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_DATA_ROOT": str(data_root),
            "PYTHON_STUB_LOG": str(log_path),
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "TASK": "libero_goal",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/preprocess/00_hdf5_reward.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"No raw LIBERO HDF5 files found under: {raw_dir}" in result.stderr
    assert "default 20_libero_dataset step installs LeRobot data" in result.stderr
    assert "scripts/reproduce/01_prepare_assets.sh" in result.stderr
    assert not log_path.exists()


def test_process_all_libero_data_dispatches_only_mainline_suite_workflow(tmp_path: Path) -> None:
    root = _project_root()
    env = os.environ.copy()
    env["DVLA_DATA_ROOT"] = str(tmp_path / "data")

    result = subprocess.run(
        [
            "bash",
            "scripts/preprocess/process_all_libero_data.sh",
            "dry_run=true",
            "tasks=[libero_goal]",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[workflow:preprocess_all] run preprocess_suite" in result.stdout
    assert "scripts/preprocess/prepare_libero_data.sh" in result.stdout
    assert "task=libero_goal" in result.stdout
    assert "pretoken" not in result.stdout.lower()
    assert "hidden_token" not in result.stdout.lower()
