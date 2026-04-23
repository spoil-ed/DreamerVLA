"""Smoke test for TSSMWorldModelTransDreamer io_mode='token'.

Builds the WM with fake config values, feeds a synthetic batch of image BPE
ids, runs forward + backward, and prints loss components.  Does NOT require
the real Chameleon checkpoint — uses `use_pretrained_backbone=False` and a
lightweight CausalTransformerCell as the dynamics backbone.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.models.world_model.tssm import TSSMWorldModelTransDreamer


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_image_tokens_vocab = 8192
    n_img = 256  # 16x16
    full_vocab = 65536
    bs = 2

    # Build a synthetic bpe2img map: image BPEs occupy ids [100, 100+8192)
    image_bpe_start = 100
    image_token_bpe_ids = torch.arange(
        image_bpe_start,
        image_bpe_start + num_image_tokens_vocab,
        dtype=torch.long,
    )

    wm = TSSMWorldModelTransDreamer(
        io_mode="token",
        token_embed_dim=64,           # tiny for smoke test
        num_image_tokens_vocab=num_image_tokens_vocab,
        hidden_dim=64,
        in_channels=64,
        obs_dim=128,
        action_dim=7,
        latent_dim=32,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        mapper_hidden_dim=128,
        reward_hidden_dim=64,
        use_pretrained_backbone=False,  # from-scratch small transformer
        freeze_transition_backbone=False,
        spatial_codec=True,
        spatial_grid=(16, 16),
        n_image_tokens=256,
        stem_init_proj_channels=48,
        stem_stage_channels=(32, 64),
        decoder_mid_channels=64,
        decoder_bspace_groups=8,
        decoder_stage_channels=(48, 24),
        decoder_stoch_hidden=64,
        free_nats=1.0,
        kl_balance=0.8,
        kl_loss_coef=1.0,
        transition_loss_coef=1.0,
        reward_loss_coef=0.0,
        image_decoder_enabled=True,
        image_decoder_loss_coef=1.0,
        image_recon_ce_coef=1.0,
        image_recon_mse_coef=0.0,
    ).to(device)

    wm.attach_lm_head(
        lm_head=None,
        image_token_bpe_ids=image_token_bpe_ids,
        full_vocab_size=full_vocab,
    )

    # Fake batch: obs / next_obs are image BPE ids in the valid image range.
    obs = torch.randint(
        image_bpe_start, image_bpe_start + num_image_tokens_vocab,
        (bs, n_img), dtype=torch.long, device=device,
    )
    next_obs = torch.randint(
        image_bpe_start, image_bpe_start + num_image_tokens_vocab,
        (bs, n_img), dtype=torch.long, device=device,
    )
    action = torch.randn(bs, 1, 7, device=device)  # [B, H=1, A]

    batch = {
        "obs_embedding": obs,
        "next_obs_embedding": next_obs,
        "action": action,
    }

    # Forward + backward
    loss_dict = wm(batch)
    print("-- forward OK --")
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            print(f"  {k:<22} = {v.item():.6f}  shape={tuple(v.shape)}")
    loss_dict["loss"].backward()
    print("-- backward OK --")

    # Check gradient flow
    n_params_with_grad = sum(
        1 for p in wm.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    n_trainable = sum(1 for p in wm.parameters() if p.requires_grad)
    print(f"  params with non-zero grad: {n_params_with_grad} / {n_trainable}")

    # Check predict_next_image_token_ids
    pred_bpe = wm.predict_next_image_token_ids(obs, action.mean(dim=1))
    print(f"-- predict_next_image_token_ids OK --  shape={tuple(pred_bpe.shape)} dtype={pred_bpe.dtype}")
    print(f"  sample[0, :8] = {pred_bpe[0, :8].tolist()}")
    assert pred_bpe.shape == (bs, n_img)
    assert pred_bpe.min() >= image_bpe_start
    assert pred_bpe.max() < image_bpe_start + num_image_tokens_vocab

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
