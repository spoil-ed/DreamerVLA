from __future__ import annotations

from pathlib import Path


def test_libero_env_layout_matches_rlinf_style() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_dir = project_root / "dreamervla" / "envs"
    libero_dir = env_dir / "libero"

    assert (env_dir / "world_model").is_dir()
    assert (libero_dir / "__init__.py").is_file()
    assert (libero_dir / "libero_env.py").is_file()
    assert (libero_dir / "utils.py").is_file()
    assert (libero_dir / "venv.py").is_file()
    assert {path.name for path in libero_dir.glob("*.py")} == {
        "__init__.py",
        "libero_env.py",
        "utils.py",
        "venv.py",
    }

    blocked_top_level = {
        "image_utils.py",
        "libero_env.py",
        "libero_chunk_env.py",
        "libero_eval_env.py",
        "libero_online_env.py",
        "online_egl_venv.py",
        "rlinf_reconfigure_venv.py",
        "rlinf_venv.py",
        "train_env.py",
    }
    present = {path.name for path in env_dir.glob("*.py")}

    assert blocked_top_level.isdisjoint(present)


def test_libero_env_exposes_chunk_as_method_not_env_type() -> None:
    from dreamervla.envs.libero import libero_env

    assert hasattr(libero_env.LiberoEnv, "step")
    assert hasattr(libero_env.LiberoEnv, "chunk_step")
    assert not hasattr(libero_env, "LiberoChunkEnv")
