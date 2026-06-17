"""Empirical probe: verify LIBERO demo field composition at t=0.

Run with:
    cd /mnt/data/spoil/workspace/DreamerVLA
    export MUJOCO_GL=osmesa
    export DVLA_DATA_ROOT="$(pwd -P)/data"
    export LIBERO_CONFIG_PATH="${DVLA_DATA_ROOT}/.libero"
    export CUDA_VISIBLE_DEVICES=0
    PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python scripts/probe_field_mapping.py
"""

from __future__ import annotations

import math
import os

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# 0. Load demo t=0 fields
# ---------------------------------------------------------------------------
DEMO_PATH = "data/datasets/libero/datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5"

print("Loading demo ...")
with h5py.File(DEMO_PATH, "r") as f:
    d = f["data"]["demo_0"]
    init_state = d.attrs["init_state"][:]
    demo_ee_pos = d["obs"]["ee_pos"][0]
    demo_ee_ori = d["obs"]["ee_ori"][0]
    demo_ee_states = d["obs"]["ee_states"][0]
    demo_gripper_states = d["obs"]["gripper_states"][0]
    demo_joint_states = d["obs"]["joint_states"][0]
    demo_robot_states = d["robot_states"][0]
    demo_states_t0 = d["states"][0]
    model_xml = d.attrs["model_file"]

print(f"  S (states dim) = {init_state.shape[0]}")
print(f"  init_state == states[0]: {np.allclose(init_state, demo_states_t0, atol=1e-10)}")

# ---------------------------------------------------------------------------
# 1. Boot LIBERO env and force demo init_state
# ---------------------------------------------------------------------------
print("\nBooting LIBERO env (libero_goal, task_id=0) ...")
from libero.libero import benchmark as libero_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = libero_benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_goal"]()
task = task_suite.get_task(0)
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)

print("  Resetting env and forcing demo init_state ...")
env.reset()
raw_obs = env.set_init_state(init_state)

print(f"  raw_obs keys: {sorted(raw_obs.keys())}")

# ---------------------------------------------------------------------------
# 2. Extract candidate fields from raw obs
# ---------------------------------------------------------------------------
eef_pos = raw_obs["robot0_eef_pos"]
eef_quat = raw_obs["robot0_eef_quat"]
gripper_qpos = raw_obs["robot0_gripper_qpos"]
joint_pos = raw_obs["robot0_joint_pos"]


def quat2axisangle(quat):
    """Quaternion (x, y, z, w) -> axis-angle (ax, ay, az)."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


candidate_ee_pos = eef_pos
candidate_ee_ori = quat2axisangle(eef_quat)
candidate_ee_states = np.concatenate([candidate_ee_pos, candidate_ee_ori])
candidate_gripper_states = gripper_qpos
candidate_joint_states = joint_pos
candidate_robot_states = np.concatenate([eef_pos, eef_quat, gripper_qpos])

# states from sim
sim_state = env.sim.get_state().flatten()

# ---------------------------------------------------------------------------
# 3. Compare each candidate against demo at t=0
# ---------------------------------------------------------------------------
ATOL = 1e-5

def check(name, candidate, demo_val, atol=ATOL):
    cand = np.array(candidate, dtype=np.float64)
    demo = np.array(demo_val, dtype=np.float64)
    match = np.allclose(cand, demo, atol=atol)
    max_err = np.max(np.abs(cand - demo)) if cand.shape == demo.shape else float("inf")
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name}: candidate={cand.shape}, demo={demo.shape}, max_err={max_err:.2e}")
    if not match:
        print(f"         candidate: {cand}")
        print(f"         demo:      {demo}")
    return match

print("\n--- Field mapping verification ---")
r1 = check("ee_pos        = robot0_eef_pos", candidate_ee_pos, demo_ee_pos)
r2 = check("ee_ori        = quat2axisangle(robot0_eef_quat)", candidate_ee_ori, demo_ee_ori)
r3 = check("ee_states     = concat(ee_pos, ee_ori)", candidate_ee_states, demo_ee_states)
r4 = check("gripper_states= robot0_gripper_qpos", candidate_gripper_states, demo_gripper_states)
r5 = check("joint_states  = robot0_joint_pos", candidate_joint_states, demo_joint_states)
r6 = check("robot_states  = concat(eef_pos, eef_quat, gripper_qpos)", candidate_robot_states, demo_robot_states)
r7 = check("states        = env.sim.get_state().flatten()", sim_state, demo_states_t0)

print(f"\nSim state dim S = {sim_state.shape[0]}")
print(f"Demo states dim S = {demo_states_t0.shape[0]}")

all_pass = all([r1, r2, r3, r4, r5, r6, r7])
print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED — adjust formulas above'}")

# Print confirmed formulas for copy-paste
print("\n--- CONFIRMED FORMULAS (copy into test) ---")
print(f"ee_pos         = raw['robot0_eef_pos']                                  # {eef_pos.shape}")
print(f"ee_ori         = quat2axisangle(raw['robot0_eef_quat'])                 # {candidate_ee_ori.shape}")
print(f"ee_states      = concat([ee_pos, ee_ori])                               # {candidate_ee_states.shape}")
print(f"gripper_states = raw['robot0_gripper_qpos']                             # {gripper_qpos.shape}")
print(f"joint_states   = raw['robot0_joint_pos']                                # {joint_pos.shape}")
print(f"robot_states   = concat([eef_pos, eef_quat, gripper_qpos])              # {candidate_robot_states.shape}")
print(f"states         = env.sim.get_state().flatten()                          # {sim_state.shape}, S={sim_state.shape[0]}")

env.close()
