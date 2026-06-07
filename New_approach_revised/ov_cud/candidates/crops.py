"""Candidate crop construction (design.md sec 6.2).

v1 produces the *box crop* only (no resize here; encoders resize to their own
input size). ``masked_crop`` and ``context_crop`` are produced only if enabled
in the config; the fields are always present in ``Candidate.crops`` (possibly
None) so later milestones can switch on 3-crop fusion without interface churn.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _clip_box(bbox: Tuple[int, int, int, int], shape) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(x1 + 1, min(x2, w)))
    y2 = int(max(y1 + 1, min(y2, h)))
    return x1, y1, x2, y2


def box_crop(image: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(bbox, image.shape)
    return image[y1:y2, x1:x2].copy()


def masked_crop(image: np.ndarray, mask: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(bbox, image.shape)
    region = image[y1:y2, x1:x2].copy()
    sub = mask[y1:y2, x1:x2]
    region[~sub] = 0
    return region


def context_crop(image: np.ndarray, bbox, expand: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)
    ex, ey = int(bw * expand), int(bh * expand)
    return box_crop(image, (x1 - ex, y1 - ey, x2 + ex, y2 + ey))


def build_crops(
    image: np.ndarray,
    mask: np.ndarray,
    bbox,
    *,
    use_masked: bool = False,
    use_context: bool = False,
    context_expand: float = 0.25,
) -> Dict[str, Optional[np.ndarray]]:
    crops: Dict[str, Optional[np.ndarray]] = {
        "box": box_crop(image, bbox),
        "masked": None,
        "context": None,
    }
    if use_masked:
        crops["masked"] = masked_crop(image, mask, bbox)
    if use_context:
        crops["context"] = context_crop(image, bbox, context_expand)
    return crops
