"""Color feature extraction."""

from typing import List

import numpy as np
from skimage.color import rgb2lab

from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def color_feature(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """不使用可调 histogram bins，使用 Lab 颜色的一/二阶统计量和分位数。"""
    lab = rgb2lab(image_rgb)
    pixels = lab[mask]

    if pixels.size == 0:
        return np.zeros(15, dtype=np.float64)

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    q25 = np.quantile(pixels, 0.25, axis=0)
    q50 = np.quantile(pixels, 0.50, axis=0)
    q75 = np.quantile(pixels, 0.75, axis=0)

    vec = np.concatenate([mean, std, q25, q50, q75]).astype(np.float64)
    return safe_normalize(vec)


def attach_color_features(image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["color"] = color_feature(image_rgb, cand.mask)
