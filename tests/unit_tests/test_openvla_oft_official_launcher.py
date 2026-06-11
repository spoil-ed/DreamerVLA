from __future__ import annotations

import re
from pathlib import Path


def test_official_openvla_oft_launcher_is_self_contained_and_portable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    launcher = project_root / "scripts" / "eval" / "launch_openvla_oft_official_libero_eval.sh"

    text = launcher.read_text(encoding="utf-8")

    assert re.search(r"^\s*source\s+.*common_env\.sh", text, re.MULTILINE) is None
    assert 'export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"' in text
    assert 'LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"' in text
    assert "datasets: ${DVLA_DATA_ROOT}/dataset/libero" in text
    assert 'CKPT_ROOT="${CKPT_ROOT:-${DVLA_DATA_ROOT}/ckpts/Openvla-oft-SFT-traj1}"' in text
    assert 'OUT_ROOT="${OUT_ROOT:-${DVLA_DATA_ROOT}/outputs/eval/openvla_oft_official_libero}"' in text
    assert 'STAGED_CKPT_ROOT="${STAGED_CKPT_ROOT:-${DVLA_DATA_ROOT}/tmp_ckpts/openvla_oft_official_eval}"' in text
    assert 'MUJOCO_GL="${MUJOCO_GL:-egl}"' in text
    assert 'PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"' in text
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
    assert 'OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${ROOT}/third_party/openvla-oft}"' in text
    assert "env DREAMERVLA_OFFICIAL_LIBERO_WORKER=1 DVLA_ROOT='${DVLA_ROOT}' DVLA_DATA_ROOT='${DVLA_DATA_ROOT}'" in text
