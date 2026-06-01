"""Parameter-free blob candidate generator.

Blob detection normally requires a response threshold. To stay parameter-free
we derive scales from image resolution and use Otsu to automatically decide
which scale-space LoG responses to keep.
"""

from typing import List

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from pf_cud.candidates.sam_candidates import mask_to_bbox
from pf_cud.data import Candidate


def image_adaptive_sigmas(h: int, w: int) -> List[float]:
    """根据图像尺寸自动生成尺度（非任务参数，而是金字塔的固定构造规则）。"""
    short = min(h, w)
    min_sigma = max(1.0, short / 512.0)
    max_sigma = max(min_sigma * 2.0, short / 32.0)
    num = int(np.ceil(np.log2(max_sigma / min_sigma + 1))) + 4
    return np.geomspace(min_sigma, max_sigma, num=num).tolist()


def disk_mask(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


class BlobCandidateGenerator:
    """参数自由 blob 候选生成器，用 Otsu 自动决定哪些 LoG 响应值得保留。"""

    def generate(self, image_rgb: np.ndarray) -> List[Candidate]:
        h, w = image_rgb.shape[:2]
        gray = rgb2gray(image_rgb)

        sigmas = image_adaptive_sigmas(h, w)

        responses = []
        for sigma in sigmas:
            # LoG 响应乘 sigma^2 做尺度归一化。
            resp = -gaussian_laplace(gray, sigma=sigma) * (sigma ** 2)
            responses.append(resp)

        responses = np.stack(responses, axis=0)  # [S, H, W]

        # 在 scale-space 中找局部最大。
        local_max = responses == maximum_filter(responses, size=(3, 3, 3))
        positive = responses > 0

        all_values = responses[positive]
        if all_values.size == 0:
            return []

        try:
            tau = threshold_otsu(all_values)
        except ValueError:
            return []

        keep = local_max & (responses >= tau)

        candidates: List[Candidate] = []
        scale_ids, ys, xs = np.where(keep)

        for sid, y, x in zip(scale_ids, ys, xs):
            sigma = sigmas[int(sid)]
            radius = np.sqrt(2) * sigma
            mask = disk_mask(h, w, y, x, radius)
            if mask.sum() == 0:
                continue
            bbox = mask_to_bbox(mask)
            candidates.append(
                Candidate(
                    mask=mask,
                    bbox=bbox,
                    source="blob",
                    score=float(responses[sid, y, x]),
                    meta={
                        "sigma": float(sigma),
                        "response": float(responses[sid, y, x]),
                    },
                )
            )

        return candidates
