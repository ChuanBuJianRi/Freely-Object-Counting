"""Shape feature extraction."""

from typing import List, Tuple

import numpy as np
from skimage.measure import moments_hu, perimeter

from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def shape_feature(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    area = float(mask.sum())
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))

    area_norm = area / float(h * w)
    aspect = box_w / box_h
    extent = area / (box_w * box_h)

    per = float(perimeter(mask))
    compactness = (4.0 * np.pi * area) / (per ** 2 + 1e-8)

    hu = moments_hu(mask.astype(float))
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    vec = np.array(
        [
            area_norm,
            np.log1p(aspect),
            extent,
            compactness,
            box_w / w,
            box_h / h,
            *hu.tolist(),
        ],
        dtype=np.float64,
    )

    return safe_normalize(vec)


def attach_shape_features(candidates: List[Candidate]) -> None:
    for cand in candidates:
        cand.features["shape"] = shape_feature(cand.mask, cand.bbox)
