from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import torch

from dreamer_vla.models.encoder import RynnVLAEncoder
from dreamer_vla.utils.paths import checkpoints_path
from dreamer_vla.utils.torch_utils import freeze_module


def load_encoder_state(encoder: RynnVLAEncoder, ckpt_path: str) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"encoder state ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("encoder")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.encoder")
    dtype = next(encoder.parameters()).dtype
    state = {
        key: value.to(dtype=dtype) if torch.is_floating_point(value) else value
        for key, value in state.items()
    }
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(
        f"[init] encoder loaded: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def build_encoder(args: argparse.Namespace, device: torch.device) -> RynnVLAEncoder:
    action_head_type = str(getattr(args, "action_head_type", None) or "legacy")
    print(
        f"[init] encoder source: model_path={args.vla_ckpt_path} "
        f"encoder_state_ckpt={getattr(args, 'encoder_state_ckpt', None) or '<none>'} "
        f"action_head_type={action_head_type}",
        flush=True,
    )
    encoder = RynnVLAEncoder(
        model_path=args.vla_ckpt_path,
        tokenizer_path=str(checkpoints_path("models--Alpha-VLLM--Lumina-mGPT-7B-768")),
        text_tokenizer_path=str(
            checkpoints_path("chameleon", "tokenizer", "text_tokenizer.json")
        ),
        chameleon_vqgan_config=str(
            checkpoints_path("chameleon", "tokenizer", "vqgan.yaml")
        ),
        chameleon_vqgan_ckpt=str(
            checkpoints_path("chameleon", "tokenizer", "vqgan.ckpt")
        ),
        resolution=256,
        action_dim=7,
        time_horizon=5,
        action_head_type=action_head_type,
        pool="mean",
        freeze_backbone=True,
    ).to(device)
    enc_ckpt = getattr(args, "encoder_state_ckpt", None)
    if enc_ckpt:
        load_encoder_state(encoder, enc_ckpt)
    else:
        print(
            f"[init] encoder built with action_head_type={action_head_type}, no separate encoder_state_ckpt",
            flush=True,
        )
    freeze_module(encoder)
    encoder.eval()
    return encoder


def load_world_model_state(
    world_model: torch.nn.Module, ckpt_path: str, reset_reward_head: bool = False
) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"world model ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("world_model") or payload.get("model")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.world_model or model")
    model_state = world_model.state_dict()
    dtype = next(world_model.parameters()).dtype
    cleaned: dict[str, torch.Tensor] = {}
    skipped_reward = 0
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        if reset_reward_head and key.startswith("reward_head."):
            skipped_reward += 1
            continue
        if key.startswith("reward_head.net.") and not key.startswith(
            "reward_head.net.net."
        ):
            candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
            if candidate in model_state:
                key = candidate
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            continue
        cleaned[key] = (
            value.to(dtype=dtype) if torch.is_floating_point(value) else value
        )
    missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
    print(
        f"[init] world_model loaded: tensors={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if skipped_reward:
        print(f"[init] skipped reward head tensors: {skipped_reward}", flush=True)


@torch.no_grad()
def obs_to_action_hidden(
    encoder: RynnVLAEncoder,
    processor: Any,
    obs: dict[str, Any],
    device: torch.device,
    target_token_id: int,
) -> torch.Tensor:
    record = obs["vla_record"]
    tokens = processor.process_item(record, training_mode=False)
    if isinstance(tokens, tuple):
        tokens = tokens[0]
    input_ids_list = [[int(tok) for tok in tokens]]
    labels = [[-100] * len(input_ids_list[0])]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*nested_from_padded CUDA kernels only support.*"
        )
        warnings.filterwarnings(
            "ignore", message=r".*PyTorch API of nested tensors is in prototype stage.*"
        )
        _, _, _, hidden_states, _, _, _ = encoder.backbone(
            input_ids=input_ids_list,
            labels=labels,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
    max_len = int(hidden_states.shape[1])
    seq = input_ids_list[0]
    row = [int(tok) for tok in seq[:max_len]] + [int(target_token_id)]
    mask = [1] * min(len(seq), max_len) + [1]
    target_len = max_len + 1
    if len(row) < target_len:
        row.extend([0] * (target_len - len(row)))
        mask.extend([0] * (target_len - len(mask)))
    input_ids = torch.tensor([row[:target_len]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask[:target_len]], dtype=torch.bool, device=device)
    action_hidden = encoder.extract_action_hidden(
        hidden_states=hidden_states,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_token_id=int(target_token_id),
        eval=True,
    )
    return action_hidden.float()


@torch.no_grad()
def obs_batch_to_action_hidden(
    encoder: RynnVLAEncoder,
    processor: Any,
    obs_batch: list[dict[str, Any]],
    device: torch.device,
    target_token_id: int,
) -> torch.Tensor:
    embeddings = [
        obs_to_action_hidden(
            encoder=encoder,
            processor=processor,
            obs=obs,
            device=device,
            target_token_id=target_token_id,
        )
        for obs in obs_batch
    ]
    return torch.cat(embeddings, dim=0)


__all__ = [
    "build_encoder",
    "load_encoder_state",
    "load_world_model_state",
    "obs_batch_to_action_hidden",
    "obs_to_action_hidden",
]
