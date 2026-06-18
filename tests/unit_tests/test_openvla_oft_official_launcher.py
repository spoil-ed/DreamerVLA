from __future__ import annotations

import re
from pathlib import Path


def test_official_openvla_oft_launcher_is_self_contained_and_portable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    launcher = project_root / "scripts" / "eval" / "launch_openvla_oft_official_libero_eval.sh"

    text = launcher.read_text(encoding="utf-8")

    assert re.search(r"^\s*source\s+.*common_env\.sh", text, re.MULTILINE) is None
    assert 'DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in text
    assert 'CKPT="${CKPT:-${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1}"' in text
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-${DVLA_DATA_ROOT}/outputs/eval/openvla_oft_official_libero}"' in text
    assert 'GPU_ID="${GPU_ID:-0}"' in text
    assert 'export MUJOCO_GL="${MUJOCO_GL:-egl}"' in text
    assert 'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"' in text
    assert 'CAMERA_INPUTS="${CAMERA_INPUTS:-primary}"' in text
    assert 'NUM_IMAGES="${NUM_IMAGES:-1}"' in text
    assert '--camera-inputs "${CAMERA_INPUTS}"' in text
    assert 'USE_PROPRIO="${USE_PROPRIO:-0}"' in text
    assert 'NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"' in text
    assert "for " not in text
    assert "case " not in text
    assert '--task-ids "${TASK_IDS}"' in text
    assert "--no-use-proprio" in text
    assert "--policy-mode" in text
    assert 'OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${DVLA_ROOT}/../WMPO/dependencies/openvla-oft}"' in text
    assert "python -m dreamervla.diagnostics.eval_openvla_oft_libero" in text
