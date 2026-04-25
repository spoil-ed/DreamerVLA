"""
Verify Chameleon VQGAN encode→decode reconstruction quality on real LIBERO frames.

Runs a set of images through:
    PIL image
      → center-crop (same as training pipeline: var_center_crop to patch grid)
      → vqgan.encode → token ids
      → vqgan.decode (codebook lookup + decoder) → PIL image

Saves a side-by-side comparison grid and prints PSNR / SSIM / L1.

Usage:
    cd /home/user01/liops/workspace/DreamerVLA
    python scripts/verify_vqgan_recon.py \
        --images /path/to/a.png /path/to/b.png \
        --out data/outputs/vqgan_recon_check
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.chameleon_model.chameleon_vae_ori.image_tokenizer import ImageTokenizer  # noqa: E402

# Default crop helper — match the training pipeline. FlexARItemProcessorActionState uses
# var_center_crop with patch_size=32 and a crop-size list derived from target_size. We
# reuse that helper so the recon matches what the world model actually sees.
from src.models.encoder.rynnvla_runtime import generate_crop_size_list, var_center_crop  # noqa: E402

VQGAN_CFG = REPO_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.yaml"
VQGAN_CKPT = REPO_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.ckpt"

DEFAULT_IMAGES = [
    REPO_ROOT
    / "data/processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_0.png",
    REPO_ROOT
    / "data/processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_100.png",
    REPO_ROOT
    / "data/processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_0.png",
    REPO_ROOT
    / "data/processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_100.png",
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    # luminance-only SSIM on float in [0,1]; keeps the script dependency-free.
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    mu_a, mu_b = af.mean(), bf.mean()
    va, vb = af.var(), bf.var()
    cov = ((af - mu_a) * (bf - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[str(p) for p in DEFAULT_IMAGES])
    ap.add_argument("--out", default=str(REPO_ROOT / "data/outputs/vqgan_recon_check"))
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    crop_list = generate_crop_size_list((args.target_size // args.patch_size) ** 2, args.patch_size)

    tok = ImageTokenizer(
        cfg_path=str(VQGAN_CFG),
        ckpt_path=str(VQGAN_CKPT),
        device=args.device,
    )

    print(f"{'image':<60}  {'PSNR(dB)':>8}  {'SSIM':>6}  {'L1':>6}")
    print("-" * 90)

    for idx, path in enumerate(args.images):
        src = Image.open(path).convert("RGB")
        src_cropped = var_center_crop(src, crop_size_list=crop_list)
        w, h = src_cropped.size
        h_lat, w_lat = h // 16, w // 16  # Chameleon VQGAN downsamples 16×

        with torch.no_grad():
            img_toks = tok.img_tokens_from_pil(src_cropped)  # tensor[h_lat*w_lat]
            recon = tok.pil_from_img_toks(img_toks, h_latent_dim=h_lat, w_latent_dim=w_lat)

        a = np.asarray(src_cropped)
        b = np.asarray(recon)
        p = psnr(a, b)
        s = ssim_simple(a, b)
        l1 = float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())
        print(f"{os.path.basename(path):<60}  {p:8.2f}  {s:6.3f}  {l1:6.2f}")

        side = Image.new("RGB", (w * 2 + 8, h), (0, 0, 0))
        side.paste(src_cropped, (0, 0))
        side.paste(recon, (w + 8, 0))
        side.save(os.path.join(args.out, f"cmp_{idx:02d}_{Path(path).stem}.png"))

    print(f"\nSaved side-by-side grids to: {args.out}")


if __name__ == "__main__":
    main()
