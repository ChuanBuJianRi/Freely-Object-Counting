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

    # perimeter() is the dominant cost because skimage scans the whole H x W
    # frame per candidate. It is translation-invariant, so computing it on the
    # bbox-local mask is numerically identical but far cheaper. moments_hu()
    # needs >=3rd-order moments and errors on tiny crops, and it is already
    # cheap (~0.5s for thousands of masks), so it stays on the full mask.
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = max(cx1 + 1, x2), max(cy1 + 1, y2)
    local = mask[cy1:cy2, cx1:cx2]

    per = float(perimeter(local))
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
