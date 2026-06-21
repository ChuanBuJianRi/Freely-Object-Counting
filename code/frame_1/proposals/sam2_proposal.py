"""SAM2 自动候选生成（OV-CUD §6.1 + §6.3）。

用冻结的 SAM2（transformers mask-generation pipeline）生成过完备 mask 候选，
再做轻量过滤：去掉过小/过大 mask、近重复 mask，并限制候选上限。

SAM2 candidate != 真实 instance，后续靠 relation / refinement / representative 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    MAX_AREA_RATIO,
    MIN_AREA_RATIO,
    MIN_BOX_SIZE,
    NEAR_DUP_IOU,
    SAM2_MODEL,
    SAM2_POINTS_PER_BATCH,
)
from ..candidates.geometry import mask_iou


@dataclass
class Candidate:
    mask: np.ndarray                       # H x W uint8
    bbox: Tuple[int, int, int, int]        # [x1, y1, x2, y2]
    area: float
    source_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class SAM2Proposer:
    """封装 SAM2 mask-generation pipeline，输出过滤后的候选列表。"""

    def __init__(
        self,
        model_name: str = SAM2_MODEL,
        device: str = "cpu",
        points_per_batch: int = SAM2_POINTS_PER_BATCH,
    ) -> None:
        from transformers import pipeline

        dev = -1 if device == "cpu" else 0
        self.generator = pipeline("mask-generation", model=model_name, device=dev)
        self.points_per_batch = points_per_batch

    def generate(
        self,
        image: np.ndarray,
        max_candidates: int = 64,
    ) -> List[Candidate]:
        """对单张图像生成候选。image: H x W x 3 uint8。"""
        from PIL import Image

        h, w = image.shape[:2]
        img_area = float(h * w)
        pil = Image.fromarray(image).convert("RGB")

        out = self.generator(pil, points_per_batch=self.points_per_batch)
        masks = out["masks"]
        scores = out.get("scores", None)
        if scores is not None:
            scores = [float(s) for s in scores]
        else:
            scores = [0.0] * len(masks)

        raw: List[Candidate] = []
        for m, s in zip(masks, scores):
            m = np.asarray(m).astype(np.uint8)
            area = float(m.sum())
            if area <= 0:
                continue
            ar = area / img_area
            if ar < MIN_AREA_RATIO or ar > MAX_AREA_RATIO:
                continue
            box = _mask_to_bbox(m)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                continue
            raw.append(Candidate(mask=m, bbox=box, area=area, source_score=s))

        # 按 source_score 降序，贪心去近重复
        raw.sort(key=lambda c: c.source_score, reverse=True)
        kept: List[Candidate] = []
        for c in raw:
            dup = False
            for k in kept:
                # bbox 快速预筛后再算 mask IoU
                if _box_iou(c.bbox, k.bbox) > 0.5 and mask_iou(c.mask, k.mask) > NEAR_DUP_IOU:
                    dup = True
                    break
            if not dup:
                kept.append(c)
            if len(kept) >= max_candidates:
                break
        return kept


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


__all__ = ["Candidate", "SAM2Proposer"]
