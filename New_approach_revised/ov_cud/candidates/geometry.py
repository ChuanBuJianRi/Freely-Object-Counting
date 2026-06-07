"""Per-candidate geometry features (design.md sec 6.2).

All features are scale-normalized so they are comparable across images. cv2 is
intentionally avoided (not installed in the target env); the perimeter for the
compactness term is computed with pure-numpy boundary counting.
"""

from __future__ import annotations

import numpy as np

# Fixed feature order (keep stable: indexed by pairwise features and the heads).
GEOMETRY_FIELDS = [
    "cx", "cy",            # normalized bbox center
    "w_norm", "h_norm",    # normalized bbox width / height
    "area_ratio_img",      # mask area / image area
    "aspect_ratio",        # bbox width / height
    "extent",              # mask area / bbox area
    "bbox_area_ratio",     # bbox area / image area
    "compactness",         # 4*pi*area / perimeter^2 in [0, 1]
]
GEOMETRY_DIM = len(GEOMETRY_FIELDS)


def _perimeter(mask: np.ndarray) -> float:
    """Count boundary pixels: True pixels with at least one False 4-neighbor."""
    if mask.sum() == 0:
        return 0.0
    m = mask
    up = np.zeros_like(m)
    down = np.zeros_like(m)
    left = np.zeros_like(m)
    right = np.zeros_like(m)
    up[1:, :] = m[:-1, :]
    down[:-1, :] = m[1:, :]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]
    # A True pixel is on the boundary if any 4-neighbor is False (or off-image).
    interior = m & up & down & left & right
    boundary = m & ~interior
    return float(boundary.sum())


def compute_geometry(mask: np.ndarray, bbox, image_shape) -> np.ndarray:
    """Return a length-``GEOMETRY_DIM`` float32 vector for one candidate."""
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    img_area = float(max(1, h_img * w_img))

    mask_area = float(mask.sum())
    bbox_area = bw * bh

    cx = ((x1 + x2) / 2.0) / max(1.0, w_img)
    cy = ((y1 + y2) / 2.0) / max(1.0, h_img)
    w_norm = bw / max(1.0, w_img)
    h_norm = bh / max(1.0, h_img)
    area_ratio_img = mask_area / img_area
    aspect_ratio = bw / bh
    extent = mask_area / bbox_area if bbox_area > 0 else 0.0
    bbox_area_ratio = bbox_area / img_area

    perim = _perimeter(mask)
    compactness = (4.0 * np.pi * mask_area) / (perim * perim) if perim > 0 else 0.0
    compactness = float(np.clip(compactness, 0.0, 1.0))

    return np.array(
        [cx, cy, w_norm, h_norm, area_ratio_img, aspect_ratio, extent,
         bbox_area_ratio, compactness],
        dtype=np.float32,
    )
