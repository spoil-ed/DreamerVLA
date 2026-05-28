from __future__ import annotations

from pathlib import Path


def test_official_openvla_oft_launcher_uses_per_task_osmesa_eval() -> None:
    project_root = Path(__file__).resolve().parents[1]
    launcher = project_root / "scripts" / "eval" / "launch_openvla_oft_official_libero_eval.sh"

    text = launcher.read_text(encoding="utf-8")

    assert 'MUJOCO_GL="${MUJOCO_GL:-osmesa}"' in text
    assert 'PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"' in text
    assert 'CAMERA_INPUTS="${CAMERA_INPUTS:-primary}"' in text
    assert 'NUM_IMAGES="${NUM_IMAGES:-${DEFAULT_NUM_IMAGES}}"' in text
    assert '--camera-inputs "${CAMERA_INPUTS}"' in text
    assert 'USE_PROPRIO="${USE_PROPRIO:-0}"' in text
    assert 'NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"' in text
    assert 'TASKS_A="${TASKS_A-0 1 2 3 4}"' in text
    assert 'TASKS_B="${TASKS_B-5 6 7 8 9}"' in text
    assert 'USE_STAGED_CKPT="${USE_STAGED_CKPT:-1}"' in text
    assert 'for tid in ${task_list}; do' in text
    assert '--task-ids "${tid}"' in text
    assert "--no-use-proprio" in text
    assert "--policy-mode" in text
