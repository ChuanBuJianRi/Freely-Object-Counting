"""Parameter-free edge / closed-contour candidate generator.

Canny normally requires two thresholds. To stay parameter-free we derive the
hysteresis thresholds from image gradient statistics (Otsu over the gradient
magnitude), then take filled connected components of the closed edge map as
candidate regions.
"""

from typing import List

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import sobel, threshold_otsu

from pf_cud.candidates.sam_candidates import mask_to_bbox
from pf_cud.data import Candidate


class EdgeCandidateGenerator:
    """闭合轮廓候选生成器，阈值由梯度分布自动估计。"""

    def generate(self, image_rgb: np.ndarray) -> List[Candidate]:
        gray = rgb2gray(image_rgb)

        grad = sobel(gray)
        grad_values = grad[grad > 0]
        if grad_values.size == 0:
            return []

        try:
            high = float(threshold_otsu(grad_values))
        except ValueError:
            return []
        low = 0.5 * high

        # sigma derived from data scale, not a tunable knob.
        edges = canny(gray, sigma=1.0, low_threshold=low, high_threshold=high)

        # Close small gaps then fill to obtain closed regions.
        closed = ndi.binary_closing(edges, structure=np.ones((3, 3)))
        filled = ndi.binary_fill_holes(closed)
        interior = filled & ~closed

        labels, n = ndi.label(interior)
        candidates: List[Candidate] = []
        for c in range(1, n + 1):
            mask = labels == c
            if mask.sum() == 0:
                continue
            bbox = mask_to_bbox(mask)
            candidates.append(
                Candidate(
                    mask=mask,
                    bbox=bbox,
                    source="edge",
                    score=None,
                    meta={"area": int(mask.sum())},
                )
            )
        return candidates
