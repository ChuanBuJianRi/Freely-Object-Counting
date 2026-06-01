"""SAM/SAM2 candidate generator.

This generator does NOT expose tunable thresholds (points_per_side,
pred_iou_thresh, stability_score_thresh, ...). It relies on the official
automatic mask generator default configuration. Over/under-segmentation is
handled downstream by the parameter-free MST/Otsu/MDL stages.
"""

from typing import List, Tuple

import numpy as np

from pf_cud.data import Candidate


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class SAMCandidateGenerator:
    """SAM/SAM2 候选生成器。

    设计要求：
    - 不暴露可调阈值。
    - 使用官方 automatic mask generator 的默认配置。
    - 过多/过少不通过手动调参解决，而通过后续 MST/Otsu/MDL 自动筛选。

    ``sam_model`` 只需暴露 ``generate(image_rgb) -> List[dict]``，每个 dict 含
    ``segmentation`` 字段（bool/0-1 mask），可选 ``predicted_iou``。
    """

    def __init__(self, sam_model):
        self.sam_model = sam_model

    def generate(self, image_rgb: np.ndarray) -> List[Candidate]:
        raw_masks = self.sam_model.generate(image_rgb)

        candidates: List[Candidate] = []
        for m in raw_masks:
            mask = np.asarray(m["segmentation"]).astype(bool)
            if mask.sum() == 0:
                continue

            bbox = mask_to_bbox(mask)
            candidates.append(
                Candidate(
                    mask=mask,
                    bbox=bbox,
                    source="sam",
                    score=m.get("predicted_iou", None),
                    meta={"raw_sam": {k: v for k, v in m.items() if k != "segmentation"}},
                )
            )

        return candidates
