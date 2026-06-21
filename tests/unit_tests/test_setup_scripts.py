from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
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
        "scripts/preprocess/validate_libero_data.sh",
    )
    for relpath in release_entrypoints:
        text = (root / relpath).read_text(encoding="utf-8")
        assert re.search(r"^\s*source\s+.*common_env\.sh", text, re.MULTILINE) is None, relpath
        assert "DVLA_ROOT" in text, relpath
        assert "DVLA_DATA_ROOT" in text, relpath

    for relpath in (
        "scripts/train_vla.sh",
        "scripts/train_wm.sh",
        "scripts/train_dreamervla.sh",
        "scripts/eval_libero_vla.sh",
    ):
        text = (root / relpath).read_text(encoding="utf-8")
        assert "dreamervla.launchers.train" in text, relpath
        assert "LIBERO_CONFIG_PATH=" not in text, relpath
    launcher_text = (root / "dreamervla" / "launchers" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "LIBERO_CONFIG_PATH" in launcher_text
    assert "datasets: {data_root}/datasets/libero" in launcher_text


def test_setup_and_download_scripts_are_release_entrypoints() -> None:
    root = _project_root()
    install = root / "scripts" / "install_env.sh"
    download = root / "scripts" / "download_assets.sh"
    install_steps = [
        root / "scripts" / "install" / name
        for name in (
            "00_apt_tools.sh",
            "10_conda_env.sh",
            "20_torch.sh",
            "30_python_deps.sh",
            "40_third_party.sh",
            "50_special_packages.sh",
            "60_verify.sh",
        )
    ]
    download_steps = [
        root / "scripts" / "download" / name
        for name in (
            "10_rynnvla.sh",
            "20_openvla_oft.sh",
            "30_openvla_oft_one_trajectory.sh",
            "40_libero_dataset.sh",
            "50_calvin_dataset.sh",
        )
    ]

    assert install.is_file()
    assert download.is_file()
    assert all(step.is_file() for step in install_steps)
    assert all(step.is_file() for step in download_steps)

    install_text = install.read_text(encoding="utf-8")
    assert "dreamervla.launchers.workflow" in install_text
    assert "--config-name install" in install_text
    assert "INSTALL_STEPS" not in install_text
    assert "run_step" not in install_text
    assert "INSTALL_ONLY" not in install_text
    assert "sudo apt" not in install_text
    assert "uv pip install" not in install_text

    step_text = "\n".join(step.read_text(encoding="utf-8") for step in install_steps)
    assert 'source "${SCRIPT_DIR}/_env.sh"' not in step_text
    assert 'DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in step_text
    assert "target conda env=" in step_text
    assert "cuda_index=" in step_text
    assert "wheel_cache=" in step_text
    assert "third_party_dir=" in step_text
    assert "optional_third_party=" in step_text
    assert "special_packages=" in step_text
    assert "checking imports in conda env=" in step_text
    assert "INSTALL_DEV_TOOLS" in step_text
    assert "apt update" in step_text
    assert "conda create -n" in step_text
    assert "uv pip install" in step_text
    assert "--group dev" in step_text
    assert "torch==2.5.1" in step_text
    assert "FLASH_ATTN" in step_text or "flash-attn" in step_text
    assert "third_party/LIBERO" in step_text
    assert "third_party/opensora" in step_text
    assert "third_party/openvla-oft" in step_text
    assert "dlimp_openvla" in step_text
    assert "transformers-openvla-oft" in step_text
    assert "egl_probe" in step_text
    verify_text = (root / "dreamervla" / "diagnostics" / "verify_install.py").read_text(
        encoding="utf-8"
    )
    assert "expected_third_party_imports" in verify_text
    assert "third_party/LIBERO" in step_text
    assert "third_party/robosuite" in step_text
    assert "third_party/robomimic" in step_text
    assert "third_party/mimicgen" in step_text

    download_text = download.read_text(encoding="utf-8")
    assert "dreamervla.launchers.workflow" in download_text
    assert "--config-name download" in download_text
    assert "DOWNLOAD_STEPS" not in download_text
    assert "DOWNLOAD_ONLY" not in download_text
    assert "10_rynnvla_chameleon.sh" not in download_text
    assert "20_lumina.sh" not in download_text
    assert "hf download" not in download_text

    download_step_text = "\n".join(step.read_text(encoding="utf-8") for step in download_steps)
    download_cfg_text = (root / "configs" / "scripts" / "download.yaml").read_text(
        encoding="utf-8"
    )
    assert 'source "${SCRIPT_DIR}/_env.sh"' not in download_step_text
    assert 'DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in download_step_text
    assert "hf download" in download_step_text
    assert "Alibaba-DAMO-Academy/WorldVLA" in download_step_text
    assert "RYNNVLA_CHAMELEON_REPO" in download_step_text
    assert "Alpha-VLLM/Lumina-mGPT-7B-768" in download_step_text
    assert "Alibaba-DAMO-Academy/RynnVLA-002" in download_step_text
    assert "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1" in download_cfg_text
    assert "OPENVLA_OFT_REPOS" in download_cfg_text
    assert "download_libero_datasets.py" in download_step_text
    assert '--download-dir "${LIBERO_DATASET_DIR}"' in download_step_text
    assert 'CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"' in download_step_text
    assert "calvin" in download_step_text.lower()
    assert "CALVIN_DOWNLOAD_METHOD" in download_step_text
    assert "VyoJ/calvin-ABCD-D-shards" in download_step_text
    assert "VyoJ/calvin-ABCD-D-subsets" in download_step_text
    assert "OpenDataLab/CALVIN" in download_step_text
    assert "HF_ENDPOINT=https://hf-mirror.com" in download_step_text


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
        "printf 'DVLA_ROOT=%s args=%s\\n' \"${DVLA_ROOT:-}\" \"$*\" >> \"${PYTHON_STUB_LOG}\"\n"
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'dreamervla.diagnostics.verify_install' ]]; then\n"
        "  test -n \"${DVLA_ROOT:-}\"\n"
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


def test_script_orchestration_is_hydra_centered() -> None:
    root = _project_root()
    workflow = root / "dreamervla" / "launchers" / "workflow.py"
    configs_dir = root / "configs" / "scripts"
    top_level = {
        "scripts/install_env.sh": "install",
        "scripts/download_assets.sh": "download",
        "scripts/preprocess_libero.sh": "preprocess_libero",
        "scripts/preprocess/prepare_libero_data.sh": "preprocess_suite",
        "scripts/preprocess/process_all_libero_data.sh": "preprocess_all",
    }

    assert workflow.is_file()
    workflow_text = workflow.read_text(encoding="utf-8")
    assert "initialize_config_dir" in workflow_text
    assert "OmegaConf" in workflow_text
    assert "subprocess.run" in workflow_text

    for relpath, config_name in top_level.items():
        text = (root / relpath).read_text(encoding="utf-8")
        assert "dreamervla.launchers.workflow" in text, relpath
        assert f"--config-name {config_name}" in text, relpath
        assert "run_step" not in text, relpath
        assert "_STEPS=(" not in text, relpath

    for name in (
        "install",
        "download",
        "preprocess_libero",
        "preprocess_suite",
        "preprocess_all",
    ):
        cfg = configs_dir / f"{name}.yaml"
        assert cfg.is_file(), name
        cfg_text = cfg.read_text(encoding="utf-8")
        assert "steps:" in cfg_text
        assert "scripts/" in cfg_text

    assert not (root / "scripts" / "install" / "_env.sh").exists()
    assert not (root / "scripts" / "download" / "_env.sh").exists()


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
    assert "ruff" not in package_names
    assert "pre-commit" not in package_names

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev_deps = set(pyproject["dependency-groups"]["dev"])
    assert {"pytest", "ruff", "pre-commit"}.issubset(dev_deps)


def test_libero_data_script_defaults_to_his1_len_action1_and_filter_noops() -> None:
    root = _project_root()
    process_all = root / "scripts" / "preprocess" / "process_all_libero_data.sh"
    prepare = root / "scripts" / "preprocess" / "prepare_libero_data.sh"
    pretokenize = root / "scripts" / "preprocess" / "20_pretokenize_dataset.sh"
    reward = root / "scripts" / "preprocess" / "10_hdf5_reward.sh"
    preprocess_cfg = (root / "configs" / "scripts" / "preprocess_suite.yaml").read_text(
        encoding="utf-8"
    )

    assert prepare.is_file()
    process_text = process_all.read_text(encoding="utf-8")
    prepare_text = prepare.read_text(encoding="utf-8")
    pretokenize_text = pretokenize.read_text(encoding="utf-8")
    reward_text = reward.read_text(encoding="utf-8")

    assert 'HIS=1' in pretokenize_text
    assert 'ACTION_HORIZON=1' in pretokenize_text
    assert 'TASK_NAME="${TASK_NAME:-${TASK}}"' in pretokenize_text
    assert 'ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"' in pretokenize_text
    assert 'ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"' in pretokenize_text
    assert "RynnVLA_LIBERO|OpenVLA_Onetraj_LIBERO" in pretokenize_text
    assert "GPUS=4,5" not in process_text
    assert 'TASK="${TASK:-libero_goal}"' in reward_text
    assert 'RAW_LIBERO_DIR="${DVLA_DATA_ROOT}/datasets/libero/${LIBERO_SUITE}"' in reward_text
    assert 'PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"' in reward_text
    assert 'marked_t_256' in reward_text
    assert 'no_noops_t_256' in reward_text
    assert "PREPROCESS_ONLY" not in prepare_text
    assert "RUN_ACTION_HIDDEN" not in prepare_text
    assert "--config-name preprocess_suite" in prepare_text
    assert "--config-name preprocess_all" in process_text
    assert "OFT_HISTORY: null" in preprocess_cfg
    assert "OFT_IMAGE_KEYS: null" in preprocess_cfg
    assert "OFT_ACTION_HIDDEN_GPUS: ${ngpu}" in preprocess_cfg
    assert "10_hdf5_reward.sh" in preprocess_cfg
    assert "20_pretokenize_dataset.sh" in preprocess_cfg
    assert "chameleon/tokenizer/vqgan.yaml" in (
        root / "scripts" / "preprocess" / "30_action_hidden.sh"
    ).read_text(encoding="utf-8")


def test_preprocess_steps_are_numbered_registered_and_individually_runnable() -> None:
    root = _project_root()
    preprocess_dir = root / "scripts" / "preprocess"
    registry = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    preprocess_cfg = (root / "configs" / "scripts" / "preprocess_suite.yaml").read_text(
        encoding="utf-8"
    )

    expected_steps = (
        "10_hdf5_reward.sh",
        "20_pretokenize_dataset.sh",
        "30_action_hidden.sh",
        "32_input_token_hidden.sh",
        "35_oft_action_hidden.sh",
        "40_validate.sh",
    )
    assert "PREPROCESS_ONLY" not in (
        preprocess_dir / "prepare_libero_data.sh"
    ).read_text(encoding="utf-8")
    numbered_steps = sorted(
        path.name
        for path in preprocess_dir.glob("[0-9][0-9]_*.sh")
        if path.is_file()
    )
    assert numbered_steps == sorted(expected_steps)
    hidden_main_docs = {"32_input_token_hidden.sh"}
    for step in expected_steps:
        script = preprocess_dir / step
        text = script.read_text(encoding="utf-8")
        assert script.is_file(), step
        assert 'source "${SCRIPT_DIR}/_env.sh"' not in text, step
        assert 'DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in text, step
        if step not in hidden_main_docs:
            assert f"`preprocess/{step}`" in registry, step
        if step in {"10_hdf5_reward.sh", "20_pretokenize_dataset.sh", "30_action_hidden.sh", "40_validate.sh"}:
            assert step in preprocess_cfg, step


def test_preprocess_scripts_are_direct_copyable_commands() -> None:
    root = _project_root()
    preprocess_dir = root / "scripts" / "preprocess"
    scripts = [
        preprocess_dir / "10_hdf5_reward.sh",
        preprocess_dir / "20_pretokenize_dataset.sh",
        preprocess_dir / "30_action_hidden.sh",
        preprocess_dir / "32_input_token_hidden.sh",
        preprocess_dir / "35_oft_action_hidden.sh",
        preprocess_dir / "40_validate.sh",
    ]

    assert not (preprocess_dir / "_env.sh").exists()
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "cmd=(" not in text, script.name
        assert "PREPROCESS_STEPS" not in text, script.name
        assert "normalize_list" not in text, script.name
        assert "${PROCESSED_DATA_ROOT:-" not in text, script.name
        assert re.search(r"\n\s*python -m ", text), script.name


def test_training_launchers_accept_common_cli_flags(tmp_path: Path) -> None:
    root = _project_root()
    log_path = tmp_path / "train_stub.log"
    python_stub = tmp_path / "python_stub.sh"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"${CUDA_VISIBLE_DEVICES:-}\" >> \"${PYTHON_STUB_LOG}\"\n"
        "printf '%s\\n' \"$*\" >> \"${PYTHON_STUB_LOG}\"\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_STUB_LOG": str(log_path),
            "DVLA_DATA_ROOT": str(tmp_path / "data"),
            "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "scripts/train_wm.sh",
            "experiment=world_model_dinowm_chunk",
            "task=libero_goal",
            "gpus=2,3",
            "ngpu=2",
            f"python={python_stub}",
            "batch_size=7",
            "num_workers=0",
            "out_dir="
            + str(tmp_path / "out"),
            "num_epochs=1",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    log_text = log_path.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2,3" in log_text
    assert "--nproc-per-node=2" in log_text
    assert "--config-name train" in log_text
    assert "experiment=world_model_dinowm_chunk" in log_text
    assert "task=libero_goal" in log_text
    assert "dataloader.batch_size=7" in log_text
    assert "dataloader.num_workers=0" in log_text
    assert f"training.out_dir={tmp_path / 'out'}" in log_text
    assert "training.num_epochs=1" in log_text


def test_training_launchers_pass_nested_task_overrides_to_train_config(tmp_path: Path) -> None:
    root = _project_root()
    hidden_dir = tmp_path / "hidden"
    result = subprocess.run(
        [
            "bash",
            "scripts/train_wm.sh",
            "experiment=oft_latent_classifier_chunk",
            "task=OpenVLA_Onetraj_LIBERO",
            f"task.openvla_oft.action_hidden_dir={hidden_dir}",
            "dry_run=true",
        ],
        cwd=root,
        env={**os.environ, "DVLA_DATA_ROOT": str(tmp_path / "data")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "task=OpenVLA_Onetraj_LIBERO" in result.stdout
    assert f"task.openvla_oft.action_hidden_dir={hidden_dir}" in result.stdout


def test_train_launcher_has_no_legacy_project_flag_mapping() -> None:
    root = _project_root()
    text = (root / "dreamervla" / "launchers" / "train.py").read_text(encoding="utf-8")

    assert "def _parse_args" not in text
    for legacy_flag in (
        "--config",
        "--task",
        "--gpus",
        "--ngpu",
        "--batch-size",
        "--num-workers",
        "--out-dir",
        "--max-steps",
        "--epochs",
        "--num-epochs",
    ):
        assert re.search(rf"(?<![\w-]){re.escape(legacy_flag)}(?![\w-])", text) is None


def test_preprocess_launchers_accept_common_cli_flags(tmp_path: Path) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    hdf5_dir = data_root / "processed_data" / "libero_goal" / "no_noops_t_256"
    hdf5_dir.mkdir(parents=True)
    (hdf5_dir / "demo.hdf5").touch()
    (
        data_root
        / "processed_data"
        / "libero_goal"
        / "tokens"
        / "libero_goal_his_1_train_third_view_wrist_w_state_1_256"
    ).mkdir(parents=True)
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'dreamervla.launchers.workflow' ]]; then\n"
        "  exec \"${REAL_PYTHON}\" \"$@\"\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"${PYTHON_STUB_LOG}\"\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_STUB_LOG": str(log_path),
            "REAL_PYTHON": sys.executable,
            "DVLA_DATA_ROOT": str(data_root),
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
            "only=[20_pretokenize_dataset]",
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
    assert "config=preprocess_suite" in workflow_output
    assert "run 20_pretokenize_dataset" in workflow_output

    dreamervla_calls = [
        line for line in log_text.splitlines() if "dreamervla.preprocess." in line
    ]
    assert dreamervla_calls
    assert any("dreamervla.preprocess.pretoken_state_action_model" in line for line in dreamervla_calls)
    assert not any(re.search(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", line) for line in dreamervla_calls)


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
        "printf '%s\\n' \"$*\" >> \"${PYTHON_STUB_LOG}\"\n"
        "module=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"${prev}\" == '-m' ]]; then module=\"${arg}\"; fi\n"
        "  prev=\"${arg}\"\n"
        "done\n"
        "case \"${module}\" in\n"
        "  dreamervla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op)\n"
        "    for arg in \"$@\"; do\n"
        "      if [[ \"${arg}\" == libero_target_dir=* ]]; then out=\"${arg#libero_target_dir=}\"; mkdir -p \"${out}\"; touch \"${out}/stub_demo.hdf5\"; fi\n"
        "    done\n"
        "    ;;\n"
        "  dreamervla.preprocess.filter_marked_libero_hdf5)\n"
        "    for arg in \"$@\"; do\n"
        "      if [[ \"${arg}\" == output_dir=* ]]; then out=\"${arg#output_dir=}\"; mkdir -p \"${out}\"; touch \"${out}/stub_demo.hdf5\"; fi\n"
        "    done\n"
        "    ;;\n"
        "  dreamervla.preprocess.preprocess_remaining_steps_reward)\n"
        "    for arg in \"$@\"; do\n"
        "      if [[ \"${arg}\" == output_dir=* ]]; then out=\"${arg#output_dir=}\"; mkdir -p \"${out}\"; touch \"${out}/stub_demo.hdf5\"; fi\n"
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
        ["bash", "scripts/preprocess/10_hdf5_reward.sh"],
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
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"${PYTHON_STUB_LOG}\"\n",
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
        ["bash", "scripts/preprocess/10_hdf5_reward.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"No raw LIBERO HDF5 files found under: {raw_dir}" in result.stderr
    assert "download.rynnvla=false download.libero=true" in result.stderr
    assert not log_path.exists()


def test_process_all_libero_data_stops_when_pretokenize_fails(tmp_path: Path) -> None:
    root = _project_root()
    data_root = tmp_path / "data"
    suite = "libero_goal"
    processed = data_root / "processed_data" / suite
    raw_dir = processed / "no_noops_t_256"
    image_dir = processed / "image_state_action_t_256"
    conv_dir = processed / "convs"
    tokenizer_dir = data_root / "checkpoints" / "models--Alpha-VLLM--Lumina-mGPT-7B-768"
    log_path = tmp_path / "python_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"

    raw_dir.mkdir(parents=True)
    (raw_dir / "demo_0.hdf5").touch()
    (image_dir / "task_0").mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.model").touch()
    conv_dir.mkdir(parents=True)
    for split in ("train", "val_ind", "val_ood"):
        (
            conv_dir
            / f"{suite}_his_1_{split}_third_view_wrist_w_state_1_256.json"
        ).write_text('[{"id": 0}]', encoding="utf-8")

    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'dreamervla.launchers.workflow' ]]; then\n"
        "  exec \"${REAL_PYTHON}\" \"$@\"\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"${PYTHON_STUB_LOG}\"\n"
        "if [[ \"${1:-}\" == '-c' ]]; then exec \"${REAL_PYTHON}\" \"$@\"; fi\n"
        "module=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"${prev}\" == '-m' ]]; then module=\"${arg}\"; break; fi\n"
        "  prev=\"${arg}\"\n"
        "done\n"
        "case \"${module}\" in\n"
        "  dreamervla.preprocess.pretoken_state_action_model) exit 42 ;;\n"
        "  dreamervla.preprocess.validate_libero_data_prep) exit 77 ;;\n"
        "  dreamervla.preprocess.concat_action_world_model_data_libero) exit 0 ;;\n"
        "  dreamervla.preprocess.concat_record) exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DVLA_DATA_ROOT": str(data_root),
            "PYTHON_STUB_LOG": str(log_path),
            "REAL_PYTHON": sys.executable,
            "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "SUITES": suite,
            "PRETOKENIZE_PROCS": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/preprocess/process_all_libero_data.sh"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 1
    assert any("dreamervla.preprocess.pretoken_state_action_model" in call for call in calls)
    assert not any("dreamervla.preprocess.validate_libero_data_prep" in call for call in calls)
    assert "[workflow:preprocess_all] run 20_pretokenize_dataset" in result.stdout
    assert "returned non-zero exit status 42" in result.stderr


def test_setup_docs_explain_libero_noop_preprocessing_order() -> None:
    root = _project_root()
    setup = (root / "SETUP.md").read_text(encoding="utf-8")

    assert "stage 1: replay and mark no-ops" in setup
    assert "keep_noops=true" in setup
    assert "stage 2: filter marked no-ops" in setup
    assert "filter_noops=true" in setup
    assert "dreamervla.preprocess.filter_marked_libero_hdf5" in setup
    assert "${DVLA_DATA_ROOT}/processed_data/${TASK}/marked_t_256" in setup
    assert "${DVLA_DATA_ROOT}/processed_data/${TASK}/no_noops_t_256" in setup
    removed_reward_dir = "${TASK}_" + "no_noops_t_256_" + "pi" + "06"
    assert removed_reward_dir not in setup


def test_setup_docs_explain_one_shot_four_suite_libero_preprocessing() -> None:
    root = _project_root()
    setup = (root / "SETUP.md").read_text(encoding="utf-8")

    assert "bash scripts/preprocess_libero.sh" in setup
    assert "libero_goal libero_object libero_spatial libero_10" in setup
    assert 'tasks=\'"libero_goal libero_object"\'' in setup


def test_top_level_preprocess_libero_wrapper_uses_repo_root_and_data_root() -> None:
    root = _project_root()
    wrapper = root / "scripts" / "preprocess_libero.sh"

    assert wrapper.is_file()
    text = wrapper.read_text(encoding="utf-8")
    cfg_text = (root / "configs" / "scripts" / "preprocess_libero.yaml").read_text(
        encoding="utf-8"
    )

    assert 'export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"' in text
    assert 'DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in text
    assert "--config-name preprocess_libero" in text
    assert "libero_goal libero_object libero_spatial libero_10" in cfg_text
    assert "scripts/preprocess/prepare_libero_data.sh" in cfg_text


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
        "check_ray.sh",
        "collect_parallel.sh",
        "download_assets.sh",
        "e2e_coldstart_warmup_cotrain_noray.sh",
        "e2e_coldstart_warmup_cotrain_ray.sh",
        "eval_libero_vla.sh",
        "install_env.sh",
        "preprocess_libero.sh",
        "start_ray.sh",
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
    path_script_re = re.compile(r"dreamervla/[^\s\"']+\.py")
    offenders = {
        str(path.relative_to(root)): path_script_re.findall(path.read_text(encoding="utf-8"))
        for path in active_shells
        if path_script_re.search(path.read_text(encoding="utf-8"))
    }

    assert offenders == {}


def test_active_shell_scripts_use_hydra_overrides_for_dreamervla_modules() -> None:
    root = _project_root()
    allowed_flags = {
        "--config-name",
        "--master-port",
        "--module",
        "--nnodes",
        "--nproc-per-node",
        "--standalone",
    }
    offenders: dict[str, list[str]] = {}

    for script in sorted((root / "scripts").rglob("*.sh")):
        if "archive" in script.relative_to(root / "scripts").parts:
            continue
        text = script.read_text(encoding="utf-8")
        commands = re.sub(r"\\\n\s*", " ", text).splitlines()
        for command in commands:
            if "dreamervla." not in command:
                continue
            if "python -m dreamervla." not in command and "--module dreamervla." not in command:
                continue
            flags = sorted(
                set(re.findall(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", command))
            )
            bad = [flag for flag in flags if flag not in allowed_flags]
            if bad:
                offenders.setdefault(str(script.relative_to(root)), []).extend(bad)

    assert offenders == {}


def test_training_launchers_do_not_nest_out_dir_default_expansion() -> None:
    root = _project_root()
    launchers = [
        root / "scripts" / "train_vla.sh",
        root / "scripts" / "train_wm.sh",
        root / "scripts" / "train_dreamervla.sh",
    ]
    offenders = [
        str(path.relative_to(root))
        for path in launchers
        if "OUT_DIR:-<config default:" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_release_scripts_are_registered() -> None:
    root = _project_root()
    scripts = root / "scripts"
    registry = (scripts / "README.md").read_text(encoding="utf-8")
    registered = set(re.findall(r"`([^`]+)`", registry))
    allowed_unregistered = {"README.md", "preprocess/32_input_token_hidden.sh"}
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
    assert "scripts/download/10_rynnvla.sh" in manifest_text
    assert "scripts/download/20_openvla_oft.sh" in manifest_text
    assert "scripts/download/30_openvla_oft_one_trajectory.sh" in manifest_text
    assert "scripts/preprocess/prepare_libero_data.sh" in manifest_text
    assert "DVLA_DATA_ROOT does not need to live inside DVLA_ROOT" in manifest_text
    assert "docs/data_layout.md" in setup
    assert "docs/data_layout.md" in readme
    assert "docs/data_layout.md" in scripts_readme


def test_release_scripts_fall_back_to_dvla_root_data() -> None:
    root = _project_root()
    old_relative_default = "${DVLA_DATA_ROOT:-" + "data}"
    active_paths = [
        *(
            path
            for path in (root / "scripts").rglob("*.sh")
            if "archive" not in path.relative_to(root / "scripts").parts
        ),
    ]
    relative_defaults = [
        str(path.relative_to(root))
        for path in active_paths
        if old_relative_default in path.read_text(encoding="utf-8")
    ]
    dvla_root_fallbacks = [
        str(path.relative_to(root))
        for path in active_paths
        if '${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}' in path.read_text(encoding="utf-8")
    ]

    assert relative_defaults == []
    assert dvla_root_fallbacks


def test_release_text_does_not_reference_removed_setup_steps() -> None:
    root = _project_root()
    checked_paths = [
        root / "README.md",
        root / "README.zh-CN.md",
        root / "SETUP.md",
        root / "docs" / "install.md",
        root / "docs" / "data_layout.md",
        root / "scripts" / "README.md",
        root / "requirements.txt",
        *(
            path
            for path in (root / "scripts").rglob("*")
            if path.is_file()
            and path.suffix in {".sh", ".md"}
            and "archive" not in path.relative_to(root / "scripts").parts
            and "__pycache__" not in path.parts
        ),
    ]
    removed_step_re = re.compile(
        r"10_worldvla\.sh|20_lumina\.sh|30_rynnvla\.sh|"
        r"20_python_deps\.sh|30_third_party\.sh|40_verify\.sh"
    )
    offenders = {
        str(path.relative_to(root)): sorted(set(removed_step_re.findall(path.read_text(encoding="utf-8"))))
        for path in checked_paths
        if removed_step_re.search(path.read_text(encoding="utf-8"))
    }

    assert offenders == {}


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
