"""Small image helpers shared by env and rollout collectors."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 image with the env's canonical bilinear rule."""
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.asarray(
        Image.fromarray(image).resize((size, size), resample=resample),
        dtype=np.uint8,
    )
