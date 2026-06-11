"""Shared image crop helpers for RynnVLA tokenization."""

from __future__ import annotations

import random

from PIL import Image


def center_crop(pil_image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    """Resize and randomly crop an image to ``crop_size``."""
    while (
        pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]
    ):
        pil_image = pil_image.resize(
            tuple(dim // 2 for dim in pil_image.size), resample=Image.BOX
        )

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(
        tuple(round(dim * scale) for dim in pil_image.size), resample=Image.BICUBIC
    )

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(
    pil_image: Image.Image,
    crop_size_list: list[tuple[int, int]],
    random_top_k: int = 1,
) -> Image.Image:
    """Choose one high-aspect-fit crop size and apply :func:`center_crop`."""
    width, height = pil_image.size
    rem_percent = [
        min(crop_width / width, crop_height / height)
        / max(crop_width / width, crop_height / height)
        for crop_width, crop_height in crop_size_list
    ]
    crop_size = random.choice(
        sorted(
            (
                (score, size)
                for score, size in zip(rem_percent, crop_size_list, strict=True)
            ),
            reverse=True,
        )[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(
    num_patches: int,
    patch_size: int,
    max_ratio: float = 4.0,
) -> list[tuple[int, int]]:
    """Return candidate crop sizes bounded by a maximum aspect ratio."""
    assert max_ratio >= 1.0
    crop_size_list: list[tuple[int, int]] = []
    width_patches, height_patches = int(num_patches), 1
    while width_patches > 0:
        if max(width_patches, height_patches) / min(width_patches, height_patches) <= max_ratio:
            crop_size_list.append(
                (width_patches * patch_size, height_patches * patch_size)
            )
        if (height_patches + 1) * width_patches <= num_patches:
            height_patches += 1
        else:
            width_patches -= 1
    return crop_size_list
