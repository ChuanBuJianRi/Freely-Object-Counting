"""Color feature extraction.

Lab first/second-order statistics plus quartiles (no tunable histogram bins).
The Lab conversion of the whole image is computed once and reused across all
candidates (instead of re-running rgb2lab per candidate), and per-candidate
statistics are gathered only inside each candidate's bbox. This is a pure
speed optimization; the produced feature is identical to the per-candidate
formulation.
"""

from typing import List

import numpy as np
from skimage.color import rgb2lab

from pf_cud.data import Candidate
from pf_cud.features.utils import safe_normalize


def _stats_from_pixels(pixels: np.ndarray) -> np.ndarray:
    """15-D Lab statistics (mean, std, q25, q50, q75), L2-normalized."""
    if pixels.size == 0:
        return np.zeros(15, dtype=np.float64)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    q25, q50, q75 = np.quantile(pixels, [0.25, 0.50, 0.75], axis=0)
    vec = np.concatenate([mean, std, q25, q50, q75]).astype(np.float64)
    return safe_normalize(vec)


def color_feature(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Single-candidate color feature (kept for API compatibility)."""
    lab = rgb2lab(image_rgb)
    return _stats_from_pixels(lab[mask])


def attach_color_features(image_rgb: np.ndarray, candidates: List[Candidate]) -> None:
    """Attach color features to all candidates, converting Lab only once.

    Per candidate we slice the Lab image to the candidate's bbox and select the
    masked pixels there, so the expensive rgb2lab runs a single time per image
    rather than once per candidate.
    """
    if not candidates:
        return
    lab = rgb2lab(image_rgb)
    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox
        sub_lab = lab[y1:y2, x1:x2]
        sub_mask = cand.mask[y1:y2, x1:x2]
        cand.features["color"] = _stats_from_pixels(sub_lab[sub_mask])
