"""RLinf-aligned standalone LIBERO rollout for OpenVLA-OFT (discrete one-traj).

This is the SHARED "RLinf-aligned action core": the exact OpenVLA-OFT / RLinf
LIBERO eval contract, in one place, so the standalone eval and the (ray /
non-ray) collectors can all drive the policy identically. The contract
(verified against RLinf's ~0.50 success_once on libero-goal):

  * single current-frame agentview image (num_images_in_input=1), no proprio
    for the discrete one-trajectory checkpoint;
  * prompt ``In: What action should the robot take to {task}?\nOut:``;
  * discrete decode -> q01/q99 unnormalize (handled by get_vla_action);
  * gripper post-process ``g = sign(2g-1) * -1`` (binarize @0.5 + invert);
  * execute the FULL 8-action chunk open-loop, then re-query;
  * initial no-op settle steps with the gripper held open.

Inference reuses the canonical OpenVLA-OFT ``get_vla_action`` via
``OpenVLAOFTObsActionPolicy`` (the proven path), driven through DreamerVLA's
own ``LIBERODreamerEnv`` so eval and collector share the same env + action core.

Run::

    conda activate dreamervla && export MUJOCO_GL=osmesa
    python -m dreamervla.runners.rlinf_libero_rollout \
      --ckpt data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
      --suite libero_goal --unnorm-key libero_goal_no_noops \
      --task-ids 0,1,2 --num-trials 3 --gpu-id 0
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

# Single source of truth for the gripper post-process, shared with the collectors.
from dreamervla.runners.oft_collect_common import process_action
from dreamervla.utils.progress import ProgressReporter


def policy_obs_from_env(obs: dict) -> dict:
    """Adapt a ``LIBERODreamerEnv`` obs to the OpenVLA-OFT obs contract.

    ``third_image`` is the agentview frame already rotated 180 deg by
    ``get_libero_image``; the discrete path consumes only ``full_image``.
    """
    return {"full_image": obs["third_image"], "state": obs["state"]}


def run_episode(policy, env, episode_id: int, num_open_loop: int = 8) -> bool:
    """Roll out one episode; return True iff the LIBERO task was solved."""
    obs = env.reset(episode_id=episode_id)
    task_description = env.task_description
    queue: deque = deque()
    done = False
    success = False
    while not done:
        if not queue:
            actions = policy(policy_obs_from_env(obs), task_description)
            queue.extend(actions[:num_open_loop])
        action = process_action(queue.popleft())
        obs, _reward, done, info = env.step(action)
        if done:
            success = bool(info.get("success", False))
    return success


def parse_ids(spec: str) -> list[int]:
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--task-ids", default="0")
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = str(Path(args.ckpt).expanduser().resolve())
    task_ids = parse_ids(args.task_ids)

    # Loads the canonical OpenVLA-OFT model + get_vla_action backend.
    # NB: from_checkpoint chdir's into the openvla-oft root; LIBERO resolves its
    # own absolute paths, so env construction below still works.
    from dreamervla.diagnostics.openvla_oft_obs_action_policy import (
        OpenVLAOFTObsActionPolicy,
    )

    policy = OpenVLAOFTObsActionPolicy.from_checkpoint(
        ckpt,
        task_suite_name=args.suite,
        gpu_id=args.gpu_id,
        policy_mode="discrete",
        num_images_in_input=args.num_images,
        use_proprio=False,
        center_crop=True,
        num_open_loop_steps=8,
        unnorm_key=args.unnorm_key,
    )

    from dreamervla.envs.libero_env import LIBERODreamerEnv

    total = 0
    succ = 0
    per_task: dict[int, float] = {}
    with ProgressReporter(
        len(task_ids) * args.num_trials, "rollout", unit="ep"
    ) as pbar:
        for tid in task_ids:
            env = LIBERODreamerEnv(
                task_suite_name=args.suite,
                task_id=tid,
                resolution=256,
                warmup_steps=args.num_steps_wait,
                seed=args.seed,
            )
            t_succ = 0
            for trial in range(args.num_trials):
                s = run_episode(policy, env, episode_id=trial)
                t_succ += int(s)
                total += 1
                succ += int(s)
                pbar.set(total)
                print(
                    f"[task {tid} trial {trial}] success={s}  running={succ}/{total}",
                    flush=True,
                )
            env.close()
            per_task[tid] = t_succ / max(1, args.num_trials)

    print("=== RLINF-ALIGNED ROLLOUT SUMMARY ===", flush=True)
    print(f"suite={args.suite}  unnorm_key={args.unnorm_key}", flush=True)
    print(f"success_once = {succ / max(1, total):.4f}  ({succ}/{total})", flush=True)
    print(f"per_task = {per_task}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
