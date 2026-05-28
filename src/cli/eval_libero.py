# ruff: noqa: E402
"""Standalone LIBERO rollout evaluation for saved VLA checkpoints.

Usage:
    conda activate wmpo
    bash scripts/eval_libero.sh \
        --ckpt_path data/outputs/vla/pretokenize_vla/checkpoints/epoch=005-train_vla_loss=1.234.ckpt \
        --task_suite libero_goal \
        --num_episodes 10 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import GenerationConfig

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)
from src.env import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
    TASK_MAX_STEPS,
)


DEFAULT_VLA_CKPT = str(PROJECT_ROOT / "data" / "ckpts" / "VLA_model_256" / "libero_10")
DEFAULT_TOKENIZER = str(
    PROJECT_ROOT / "data" / "ckpts" / "models--Alpha-VLLM--Lumina-mGPT-7B-768"
)
DEFAULT_TEXT_TOKENIZER = str(
    PROJECT_ROOT / "data" / "ckpts" / "chameleon" / "tokenizer" / "text_tokenizer.json"
)
DEFAULT_VQGAN_CFG = str(
    PROJECT_ROOT / "data" / "ckpts" / "chameleon" / "tokenizer" / "vqgan.yaml"
)
DEFAULT_VQGAN_CKPT = str(
    PROJECT_ROOT / "data" / "ckpts" / "chameleon" / "tokenizer" / "vqgan.ckpt"
)


def unnorm_action(action: np.ndarray) -> np.ndarray:
    min_values = np.array(
        [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
    )
    max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
    if action.shape[0] > 7:
        action = action[:7]
    return (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values


def unnorm_actions(actions: np.ndarray) -> np.ndarray:
    min_values = np.array(
        [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
    )
    max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
    if actions.ndim == 2 and actions.shape[1] > 7:
        actions = actions[:, :7]
    return (actions + 1) / 2 * (max_values - min_values + 1e-8) + min_values


def generate_actions(
    backbone,
    item_processor,
    cur_img,
    cur_wrist_img,
    state,
    task_description,
    action_steps,
    device,
):
    img_c = [cur_img, cur_wrist_img]
    human_val = (
        f"Finish the task: {task_description}."
        + "<|state|>" * 1
        + "<|image|>" * len(img_c)
    )

    conv = {
        "conversations": [{"from": "human", "value": human_val}],
        "image": img_c,
        "action": [],
        "state": [state],
    }
    tokens = item_processor.process_item(conv, training_mode=False)
    if isinstance(tokens, tuple):
        tokens = tokens[0]

    input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)

    generation_config = GenerationConfig(
        max_new_tokens=1,
        max_length=backbone.config.max_position_embeddings,
        temperature=1,
        top_k=None,
        do_sample=False,
        eos_token_id=[8710],
    )

    if hasattr(backbone, "generate_action_head"):
        try:
            predicted = backbone.generate_action_head(input_ids, generation_config)
            actions = unnorm_actions(predicted.cpu().float().detach().numpy())
            return [actions[i] for i in range(actions.shape[0])]
        except Exception as e:
            print(f"  generate_action_head failed: {e}")

    generation_config_ma = GenerationConfig(
        max_new_tokens=action_steps * 12,
        max_length=backbone.config.max_position_embeddings,
        temperature=1,
        top_k=None,
        do_sample=False,
        eos_token_id=[8710],
    )
    if hasattr(backbone, "generate_dis_ma"):
        try:
            action_sequences = backbone.generate_dis_ma(input_ids, generation_config_ma)
            results = []
            for seq in action_sequences:
                a = (
                    seq.cpu().float().detach().numpy()
                    if isinstance(seq, torch.Tensor)
                    else np.asarray(seq, dtype=np.float32)
                )
                if a.shape[0] == 7:
                    results.append(unnorm_action(a))
            return results
        except Exception as e:
            print(f"  generate_dis_ma failed: {e}")

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Standalone LIBERO eval for VLA checkpoints"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to training checkpoint (.ckpt). If None, eval the initial pretrained model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_VLA_CKPT,
        help="Pretrained model directory (HF format)",
    )
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--text_tokenizer_path", type=str, default=DEFAULT_TEXT_TOKENIZER
    )
    parser.add_argument("--vqgan_cfg", type=str, default=DEFAULT_VQGAN_CFG)
    parser.add_argument("--vqgan_ckpt", type=str, default=DEFAULT_VQGAN_CKPT)
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        choices=list(TASK_MAX_STEPS.keys()),
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Episodes per task"
    )
    parser.add_argument("--action_steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    backbone = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Load checkpoint weights if provided
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path} ...")
        payload = torch.load(args.ckpt_path, map_location="cpu")
        if "state_dicts" in payload and "encoder" in payload["state_dicts"]:
            backbone.load_state_dict(payload["state_dicts"]["encoder"], strict=False)
            print("  Loaded encoder state_dict from checkpoint.")
        else:
            print("  Warning: no 'encoder' state_dict found in checkpoint.")

    backbone = backbone.to(device).eval()

    # Build item processor
    from src.models.encoder.rynnvla_runtime import FlexARItemProcessorActionState

    item_processor = FlexARItemProcessorActionState(
        tokenizer_path=args.tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        vqgan_cfg_path=args.vqgan_cfg,
        vqgan_ckpt_path=args.vqgan_ckpt,
        target_size=args.resolution,
        device=str(device),
    )

    # Run LIBERO eval
    from libero.libero import benchmark as libero_benchmark

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS[args.task_suite]

    total_episodes, total_successes = 0, 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.resolution)

        task_successes = 0
        for episode_idx in range(min(args.num_episodes, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            done = False
            actions_buffer = []

            for t in range(max_steps + 10):
                if t < 10:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    continue

                img = get_libero_image(obs, args.resolution)
                wrist_img = get_libero_image(
                    obs, args.resolution, "robot0_eye_in_hand_image"
                )
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                if len(actions_buffer) == 0:
                    actions_buffer = generate_actions(
                        backbone,
                        item_processor,
                        Image.fromarray(img),
                        Image.fromarray(wrist_img),
                        state,
                        task_description,
                        args.action_steps,
                        device,
                    )

                if len(actions_buffer) == 0:
                    break
                action = actions_buffer.pop(0)
                obs, _, done, _ = env.step(action.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            total_episodes += 1
        env.close()

        rate = task_successes / max(args.num_episodes, 1)
        print(
            f"Task {task_id} ({task_description}): {rate:.1%} ({task_successes}/{args.num_episodes})"
        )

    avg = total_successes / max(total_episodes, 1)
    print(f"\nOverall: {avg:.1%} ({total_successes}/{total_episodes})")


if __name__ == "__main__":
    main()
