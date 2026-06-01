"""Spatial feature extraction."""

from typing import List, Tuple

import numpy as np

from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def spatial_feature(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    vec = np.array([cx, cy, bw, bh, center_dist], dtype=np.float64)
    return safe_normalize(vec)


def attach_spatial_features(candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["spatial"] = spatial_feature(cand.mask, cand.bbox)
