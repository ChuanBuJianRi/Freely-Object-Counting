"""Feature extraction helpers."""

from typing import Tuple

import numpy as np
from PIL import Image


def crop_candidate(
    image_rgb: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # 背景置零，减少背景干扰。
    crop[~crop_mask] = 0

    return Image.fromarray(crop)


def safe_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm
